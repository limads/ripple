// To perform DCT over image:
// (1) Apply DCT over image considering it as a vector (over rows);
// (2) Apply DCT over transpose of result of (1) (by making a matrix transpose) considering it as a vector (over columns).
// To apply inverse, repeat the same process.

use rustdct::DctPlanner;
use nalgebra::{Scalar, DMatrixSliceMut, DVectorSliceMut, DMatrixSlice};
use std::sync::Arc;
use rustdct::{Dct2, Dct3, TransformType2And3};
use std::iter::FromIterator;

pub struct DCT {

    planner : DctPlanner<f32>,

    sz : usize,

    fwd : Arc<dyn TransformType2And3<f32>>,

    bwd : Arc<dyn TransformType2And3<f32>>

}

pub (crate) fn scale_slice(s : &mut [f32], by : f32) {
    DVectorSliceMut::from_slice(s, s.len()).scale_mut(by);
}

// nrow and ncol are the original matrix dimensions.
pub (crate) fn transpose_slice<N>(from : &[N], (nrow, ncol) : (usize, usize), to : &mut [N])
where
    N : Scalar + Copy
{
    DMatrixSlice::from_slice(from, nrow, ncol)
        .transpose_to(&mut DMatrixSliceMut::from_slice(to, ncol, nrow));
}

impl DCT {

    pub fn new(sz : usize) -> Self {
        let mut planner = DctPlanner::<f32>::new();
        let fwd = planner.plan_dct2(sz);
        let bwd = planner.plan_dct3(sz);
        Self { planner, fwd, bwd, sz }
    }

    pub fn forward(&mut self, content : &[f32], out : &mut [f32]) {
        out.copy_from_slice(content);
        self.fwd.process_dct2(out);
    }

    pub fn backward(&mut self, content : &[f32], out : &mut [f32]) {
        out.copy_from_slice(content);
        self.bwd.process_dct3(out);
        scale_slice(out, 1. / (self.sz as f32 / 2.));
    }

}

pub struct DCT2D {

    row_dct : DCT,

    col_dct : DCT,

    buf : Option<Vec<f32>>,

    nrow : usize,

    ncol : usize

}

impl DCT2D {

    pub fn new(nrow : usize, ncol : usize) -> Self {
        // row_dct (applied over rows) has as its size the number of columns. The same for
        // the other col_dct (applied over cols with the size the number of rows).
        let row_dct = DCT::new(ncol);
        let col_dct = DCT::new(nrow);
        let buf = Vec::from_iter((0..(nrow*ncol)).map(|_| 0. ));
        Self { row_dct, col_dct, nrow, ncol, buf : Some(buf) }
    }

    pub fn forward(&mut self, content : &[f32], out : &mut [f32]) {
        let mut buf = self.buf.take().unwrap();
        buf.copy_from_slice(content);

        // Apply DCT2 over rows
        for out_row in buf.chunks_mut(self.ncol) {
            self.row_dct.fwd.process_dct2(out_row);
        }

        // Apply DCT2 over cols
        // DMatrixSliceMut::from_slice(&mut buf[..], self.nrow, self.ncol)
        //    .transpose_to(&mut DMatrixSliceMut::from_slice(out, self.ncol, self.nrow));
        transpose_slice(&buf[..], (self.nrow, self.ncol), out);
        for out_col in out.chunks_mut(self.nrow) {
            self.col_dct.fwd.process_dct2(out_col);
        }

        // Make spectrum upright again. This allows accessing horizontal frequencies with horizontal
        // indices, and vertical frequencies with vertical indices.
        buf.copy_from_slice(out);
        //DMatrixSliceMut::from_slice(&mut buf, self.ncol, self.nrow)
        //    .transpose_to(&mut DMatrixSliceMut::from_slice(out, self.nrow, self.ncol));
        transpose_slice(&buf[..], (self.ncol, self.nrow), out);

        self.buf = Some(buf);
    }

    pub fn backward(&mut self, content : &[f32], out : &mut [f32]) {
        let mut buf = self.buf.take().unwrap();

        // Make spectrum non-upright again. This returns the spectrum to the state before the
        // last transpose at forward (i.e. rows and columns switched after the first tranpose).
        // out.copy_from_slice(&content);
        //DMatrixSliceMut::from_slice(out, self.nrow, self.ncol)
        //    .transpose_to(&mut DMatrixSliceMut::from_slice(&mut buf[..], self.ncol, self.nrow));
        transpose_slice(&content[..], (self.nrow, self.ncol), &mut buf[..]);

        for out_col in buf.chunks_mut(self.nrow) {
            self.col_dct.bwd.process_dct3(out_col);
            scale_slice(out_col, 1. / ((self.nrow) as f32 / 2.));
        }
        // DMatrixSliceMut::from_slice(&mut buf[..], self.ncol, self.nrow)
        //    .transpose_to(&mut DMatrixSliceMut::from_slice(out, self.nrow, self.ncol));
        transpose_slice(&buf[..], (self.ncol, self.nrow), out);

        for out_row in out.chunks_mut(self.ncol) {
            self.row_dct.bwd.process_dct3(out_row);
            scale_slice(out_row, 1. / ((self.ncol) as f32 / 2.));
        }

        self.buf = Some(buf);
    }

}


