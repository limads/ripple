use nalgebra::{DMatrixSliceMut, DVectorSliceMut, DMatrixSlice};
use std::sync::Arc;
use std::iter::FromIterator;
use crate::dct::{scale_slice, transpose_slice};
use std::mem;
use rustfft::{FftPlanner, Fft, num_complex};

pub struct FFT {

    planner : FftPlanner<f32>,

    sz : usize,

    fwd : Arc<dyn Fft<f32>>,

    bwd : Arc<dyn Fft<f32>>,

    buf : Vec<num_complex::Complex<f32>>

}

fn copy_real_to_complex(cplx : &mut [num_complex::Complex<f32>], real : &[f32]) {
    cplx.iter_mut().zip(real.iter()).for_each(|(c, r)| *c = num_complex::Complex::new(*r, 0.) );
}

fn copy_complex_to_real(real : &mut [f32], cplx : &[num_complex::Complex<f32>]) {
    real.iter_mut().zip(cplx.iter()).for_each(|(r, c)| *r = c.re );
}

impl FFT {

    pub fn new(sz : usize) -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let fwd = planner.plan_fft_forward(sz);
        let bwd = planner.plan_fft_inverse(sz);
        let buf = Vec::from_iter((0..sz).map(|_| num_complex::Complex::new(0., 0.) ));
        Self { planner, fwd, bwd, sz, buf }
    }

    pub fn forward(&mut self, content : &[f32], out : &mut [num_complex::Complex<f32>]) {
        copy_real_to_complex(out, content);
        self.fwd.process(out);
    }

    pub fn backward(&mut self, content : &[num_complex::Complex<f32>], out : &mut [f32]) {
        let mut buf = mem::take(&mut self.buf);
        buf.copy_from_slice(content);
        self.bwd.process(&mut buf[..]);
        copy_complex_to_real(out, &buf[..]);
        self.buf = buf;
        // scale_slice(out, 1. / (self.sz as f32 / 2.));
    }

}

pub struct FFT2D {

    row_fft : FFT,

    col_fft : FFT,

    buf : Option<Vec<num_complex::Complex<f32>>>,

    bwd_buf : Option<Vec<num_complex::Complex<f32>>>,

    nrow : usize,

    ncol : usize

}

impl FFT2D {

    pub fn new(nrow : usize, ncol : usize) -> Self {
        // row_dct (applied over rows) has as its size the number of columns. The same for
        // the other col_dct (applied over cols with the size the number of rows).
        let row_fft = FFT::new(ncol);
        let col_fft = FFT::new(nrow);
        let buf = Vec::from_iter((0..(nrow*ncol)).map(|_| num_complex::Complex::new(0., 0.) ));
        let bwd_buf = Some(buf.clone());
        Self { row_fft, col_fft, nrow, ncol, buf : Some(buf), bwd_buf }
    }

    pub fn forward(&mut self, content : &[f32], out : &mut [num_complex::Complex<f32>]) {
        let mut buf = self.buf.take().unwrap();
        copy_real_to_complex(&mut buf[..], content);

        // Apply FFT2 over rows
        for out_row in buf.chunks_mut(self.ncol) {
            self.row_fft.fwd.process(out_row);
        }

        // Apply FFT2 over cols
        // DMatrixSliceMut::from_slice(&mut buf[..], self.nrow, self.ncol)
        //     .transpose_to(&mut DMatrixSliceMut::from_slice(out, self.ncol, self.nrow));
        transpose_slice(&buf[..], (self.nrow, self.ncol), out);

        for out_col in out.chunks_mut(self.nrow) {
            self.col_fft.fwd.process(out_col);
        }

        // Make spectrum upright again. This allows accessing horizontal frequencies with horizontal
        // indices, and vertical frequencies with vertical indices.
        buf.copy_from_slice(out);

        // DMatrixSliceMut::from_slice(&mut buf, self.ncol, self.nrow)
        //    .transpose_to(&mut DMatrixSliceMut::from_slice(out, self.nrow, self.ncol));
        transpose_slice(&buf[..], (self.ncol, self.nrow), out);

        self.buf = Some(buf);
    }

    pub fn backward(&mut self, content : &[num_complex::Complex<f32>], out : &mut [f32]) {
        let mut buf = self.buf.take().unwrap();
        let mut bwd_buf = self.bwd_buf.take().unwrap();

        // Make spectrum non-upright again. This returns the spectrum to the state before the
        // last transpose at forward (i.e. rows and columns switched after the first tranpose).
        // out.copy_from_slice(&content);
        // bwd_buf.copy_from_slice(content);

        //DMatrixSliceMut::from_slice(out, self.nrow, self.ncol)
        //    .transpose_to(&mut DMatrixSliceMut::from_slice(&mut buf[..], self.ncol, self.nrow));
        transpose_slice(&content[..], (self.nrow, self.ncol), &mut buf[..]);

        for out_col in buf.chunks_mut(self.nrow) {
            self.col_fft.bwd.process(out_col);
            // scale_slice(out_col, 1. / ((self.nrow) as f32 / 2.));
        }
        // DMatrixSliceMut::from_slice(&mut buf[..], self.ncol, self.nrow)
        //    .transpose_to(&mut DMatrixSliceMut::from_slice(out, self.nrow, self.ncol));
        transpose_slice(&buf[..], (self.ncol, self.nrow), &mut bwd_buf);

        for out_row in bwd_buf.chunks_mut(self.ncol) {
            self.row_fft.bwd.process(out_row);
            // scale_slice(out_row, 1. / ((self.ncol) as f32 / 2.));
        }
        copy_complex_to_real(out, &bwd_buf[..]);

        self.buf = Some(buf);
        self.bwd_buf = Some(bwd_buf);
    }

}


