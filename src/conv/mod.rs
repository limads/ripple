/// General-purpose iterators over dynamic matrices.

use nalgebra::Scalar;
use std::ops::{Mul, AddAssign, Div, DivAssign, Add};
use num_traits::identities::{Zero, One};
use std::cmp::{Eq, PartialEq};
use num_traits::Float;
use std::any::Any;
use crate::signal::Signal;

// pub mod iter;

/// Trait implemented by types which can be convolved with another instance
/// of themselves. Self must satisfy clone because the convolve(.) implementation
/// is provided by calling convolve_mut on a cloned instance.
pub trait Convolve
// where
//    Self : Clone
{

    type Output;

    fn convolve_mut(&self, filter : &Self, out : &mut Self::Output);

    /*fn convolve(&self, filter : &Self) -> Self::Output {
        let mut out = self.clone();
        self.convolve_mut(filter, &mut out);
        out
    }*/

}

// Native discrete convolution.
// pub mod native;

/// Wrapper type to perform discrete convolution by binding against Intel MKL.
#[cfg(feature = "mkl")]
pub mod mkl;

/// Convolution behavior at signal boundaries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Extension {

    // Repeat with the first and last element so ouput.len() == signal.len()
    Pad,

    // Wraps with first elements so output.len() == signal.len()
    Wrap,

    // Writes to signal.len() - (kernel.len() - 1 )
    Ignore

}

/// Baseline implementation using Rust standard library operations.
fn baseline_convolution<S : Scalar>(
    signal : &[S],
    kernel : &[S],
    out : &mut [S],
    ext : Extension
) where
    S : Scalar + Mul<Output = S> + Zero + AddAssign + Copy
{
    // out.clear();

    if ext == Extension::Ignore {
        assert!(out.len() == signal.len() - (kernel.len() - 1));
    }

    let k_len = kernel.len();
    signal.windows(k_len).enumerate().for_each(|(w_ix, win)| {
        let mut dot = S::zero();
        for s_ix in 0..k_len {
            dot += win[s_ix] * kernel[k_len-s_ix-1];
        }
        out[w_ix] = dot;
    });

    /*match padding {
        Padding::Extend => {
            while out.len() != signal.len() {
                out.push(out[out.len() - 1]);
            }
        },
        _ => unimplemented!()
    }*/
}

/*impl<'a, S> Convolve for &'a [S]
where
    S : Scalar + Mul<Output = S> + Zero + AddAssign + Copy + Any
{

    type Output = &'a mut [S];

    fn convolve_mut(&self, filter : &Self, out : Self::Output) {

        #[cfg(feature="mkl")]
        {
            if (&self[0] as &dyn Any).is::<f32>() {
            // Dispatch to MKL impl
            }

            if (&self[0] as &dyn Any).is::<f64>() {
                // Dispatch to MKL impl
            }
        }

        baseline_convolution(self.as_ref(), filter.as_ref(), out.as_mut(), Extension::Ignore);
    }

}*/

impl<S> Convolve for Signal<S>
where
    S : Scalar + Mul<Output = S> + Zero + AddAssign + Copy + Any
{

    type Output = Signal<S>;

    fn convolve_mut(&self, filter : &Self, out : &mut Self::Output) {
        let input : &[S] = self.as_ref();
        let kernel : &[S] = filter.as_ref();
        let output : &mut [S] = out.as_mut();
        baseline_convolution(self.as_ref(), filter.as_ref(), out.as_mut(), Extension::Ignore);
        // input.convolve_mut(kernel, output);
    }

}

/// Common time-domain representations of filters.
pub mod kernel {

    use super::*;

    /// Returns the finite approximation to the identity of the convolution operation.
    pub fn delta<S, const U : usize>() -> [S; U]
    where
        S : Scalar + Zero + One + Copy
    {
        let mut delta = [S::zero(); U];
        delta[U / 2] = S::one();
        delta
    }

    // Exponential decay filter.
    pub fn exponential<S, const U : usize>() -> [S; U]
    where
        S : Scalar + Zero + One + Copy + Float + Div<Output=S> + DivAssign + From<f32> + Add<Output=S>
    {
        let mut exp = [S::zero(); U];
        let mut v : S = From::from(1.0);
        assert!(U % 2 != 0);
        let center = (U-1)/2;
        exp[center] = v;
        for ix in 1..(center+1) {
            v /= From::from(2.);
            exp[center-ix] = v;
            exp[center+ix] = v;
        }
        let sum = exp.iter().fold(S::zero(), |s, a| s + *a );
        exp.iter_mut().for_each(|s| *s /= sum );
        exp
    }

    #[test]
    fn expfilter() {
        let s = exponential::<f32, 5>();
        println!("{:?} {}", s, s.iter().sum::<f32>());
    }

}

// IppStatus ippsCrossCorrNorm_32f (const Ipp32f* pSrc1, int src1Len, const Ipp32f* pSrc2,
// int src2Len, Ipp32f* pDst, int dstLen, int lowLag, IppEnum algType, Ipp8u* pBuffer );

// IppStatus ippsConvolve_32f (const Ipp32f* pSrc1, int src1Len, const Ipp32f* pSrc2, int
// src2Len, Ipp32f* pDst, IppEnum algType, Ipp8u* pBuffer );

// IppStatus ippsFIRSR_32f(const Ipp32f* pSrc, Ipp32f* pDst, int numIters,
// IppsFIRSpec_32f* pSpec, const Ipp32f* pDlySrc, Ipp32f* pDlyDst, Ipp8u* pBuf );

// IppStatus ippsFIRSparse_32f(const Ipp32f* pSrc, Ipp32f* pDst, int len,
// IppsFIRSparseState_32f* pState );

// IppStatus ippsIIR32f_16s_Sfs(const Ipp16s* pSrc, Ipp16s* pDst, int len,
// IppsIIRState32f_16s* pState, int scaleFactor );

// IppStatus ippsFilterMedian_8u(const Ipp8u* pSrc, Ipp8u* pDst, int len, int maskSize,
// const Ipp8u* pDlySrc, Ipp8u* pDlyDst, Ipp8u* pBuffer );

/*

Convolution implementation using nalgebra

use super::iter::*;
use nalgebra::*;

pub trait Convolve {

    fn convolve(&self, kernel : &Self) -> Self;

}

impl<N> Convolve for DMatrix<N>
    where
        N : Scalar + std::ops::Mul + simba::scalar::ClosedMul<N> +
            simba::scalar::Field + simba::scalar::SupersetOf<f64>,
        DMatrix<N> : WindowIterate<N, VecStorage<N, Dynamic, Dynamic>>
{

    fn convolve(&self, kernel : &Self) -> Self {
        self.windows(kernel.shape())
            .pool(|win| win.component_mul(&kernel).sum())
    }
}

use nalgebra::*;
use nalgebra::storage::*;

/// Structure that encapsulates the logic of running windows over a matrix.
/// It holds information such as the window size, the step,
/// and whether the ordering is row-wise or column-wise. The user only uses
/// the specializations of this structure returned by WindowIterate::windows()
/// and ChunkIterate::chunks() that are implemented for DMatrix<N>, that
/// pick a step size at compile time (for contiguous overlapping windows in the first case
/// and contiguous non-overlapping windows in the second case). The current definition always
/// slides over a matrix in the row direction first.
pub struct PatchIterator<'a, N, S, W, V>
    where
        N : Scalar,
        //S : ContiguousStorage<N, Dynamic, Dynamic>,
        S : Storage<N, Dynamic, Dynamic>,
        W : Dim,
        V : Dim
{
    source : &'a Matrix<N, Dynamic, Dynamic, S>,
    size : (usize, usize),

    /// Patch iterator differs from a standard Vec iterator
    /// simply by having a internal counter over two dimensions:
    /// the first count the current rows and the last count the
    /// current column. Different implementations of patch iterator
    /// just differ by the rule by which those quantities evolve at each
    /// iteration.
    curr_pos : (usize, usize),

    /// Unused for now.
    _c_stride : usize,

    /// Vertical increment. Either U1 or Dynamic.
    step_v : V,

    /// Horizontal increment. Either U1 or Dynamic.
    step_h : W,

    /// Unused for now. Assume row-wise.
    _row_wise : bool,

    /// A pooling operation very often follows iterating over a matrix.
    /// pool_dims keep the dimensions of the resulting pooling, so the
    /// pool method can be generic, while the pooling calculation logic
    /// is different for each implementation.
    pool_dims : (usize, usize)
}

impl<'a, N, S, W, V> PatchIterator<'a, N, S, W, V>
    where
        N : Scalar,
        //S : ContiguousStorage<N, Dynamic, Dynamic>,
        S : Storage<N, Dynamic, Dynamic>,
        W : Dim,
        V : Dim
{

    /// PatchIterator::pool consumes the structure, iterate over its source matrix
    /// and generates another owned matrix by applying any function that consumes a matrix
    /// and returns a scalar, such as max(), sum() and norm(). If the original matrix has
    /// dimensions (r,c) and the window size asked by the user has dimensions (wr, wc),
    /// the resulting matrix has dimensions (r/wr, c/wc).
    pub fn pool<F>(
        mut self,
        mut f : F
    ) -> Matrix<N, Dynamic, Dynamic, VecStorage<N, Dynamic, Dynamic>>
        where
            F : FnMut(Matrix<N, Dynamic, Dynamic, SliceStorage<'a, N, Dynamic, Dynamic, S::RStride, S::CStride>>)->N
    {
        //println!("Pool dims: {:?}", self.pool_dims);
        let mut data : Vec<N> = Vec::with_capacity(self.pool_dims.1 * self.pool_dims.0);

        // Since iteration is row-wise, we need to transpose the matrix.
        while let Some(w) = self.next() {
            let s = f(w);
            data.push(s);
        }
        //println!("Data size: {:?}", data.len());
        let mut ans = DMatrix::<N>::from_vec(self.pool_dims.1, self.pool_dims.0, data);
        ans.transpose_mut();
        ans
    }
}

/// Returns a WindowIterator over overlapping contiguous regions with step size 1
pub trait WindowIterate<N, S>
    where
        N : Scalar,
        //S : ContiguousStorage<N, Dynamic, Dynamic>
        S : Storage<N, Dynamic, Dynamic>,
{
    fn windows(&self, win_sz : (usize, usize)) -> PatchIterator<N, S, U1, U1>;
}

/// Returns a WindowIterator over non-overlapping contiguous regions
pub trait ChunkIterate<N, S>
    where
        N : Scalar,
        //S : ContiguousStorage<N, Dynamic, Dynamic>,
        S : Storage<N, Dynamic, Dynamic>,
{
    fn chunks(&self, sz : (usize, usize)) -> PatchIterator<N, S, Dynamic, Dynamic>;

}

impl<'a, N, S, W, V> Iterator for PatchIterator<'a, N, S, W, V>
    where
        N : Scalar,
        //S : ContiguousStorage<N, Dynamic, Dynamic>,
        S : Storage<N, Dynamic, Dynamic>,
        W : Dim,
        V : Dim
    {

    type Item = Matrix<N, Dynamic, Dynamic, SliceStorage<'a, N, Dynamic, Dynamic, S::RStride, S::CStride>>;

    fn next(&mut self) -> Option<Self::Item> {
        let win = if self.curr_pos.0  + self.size.0 <= self.source.nrows() && self.curr_pos.1 + self.size.1 <= self.source.ncols() {
            //println!("Matrix slice: pos : {:?} slice : {:?} size: {:?}", self.curr_pos, self.size, self.source.shape());
            Some(self.source.slice(self.curr_pos, self.size))
        } else {
            None
        };
        self.curr_pos.1 += self.step_h.value(); // self.size.1 for chunks; 1 for window
        if self.curr_pos.1 + self.size.1 > self.source.ncols() { // >=
            self.curr_pos.1 = 0;
            self.curr_pos.0 += self.step_v.value();
        }
        win
    }

}

impl<N, S> WindowIterate<N, S> for Matrix<N, Dynamic, Dynamic, S>
    where
        N : Scalar,
        //S : ContiguousStorage<N, Dynamic, Dynamic>
        S : Storage<N, Dynamic, Dynamic>,
{
    fn windows(
        &self,
        sz : (usize, usize)
    ) -> PatchIterator<N, S, U1, U1> {
        if self.nrows() % sz.0 != 0 || self.ncols() % sz.1 != 0 {
            panic!("Matrix size should be a multiple of window size");
        }
        let pool_dims = (self.nrows() - sz.0 + 1, self.ncols() - sz.1 + 1);
        PatchIterator::<N, S, U1, U1> {
            source : &self,
            size : sz,
            curr_pos : (0, 0),
            _c_stride : self.nrows(),
            step_h : U1{},
            step_v : U1{},
            _row_wise : false,
            pool_dims
        }
    }
}

impl<N, S> ChunkIterate<N, S> for Matrix<N, Dynamic, Dynamic, S>
    where
        N : Scalar,
        //S : ContiguousStorage<N, Dynamic, Dynamic>
        S : Storage<N, Dynamic, Dynamic>,
{
    fn chunks(
        &self,
        sz : (usize, usize)
    ) -> PatchIterator<N, S, Dynamic, Dynamic> {
        let step_v = Dim::from_usize(sz.0);
        let step_h = Dim::from_usize(sz.1);
        //println!("matrix size: {:?}; window size: {:?}", self.shape(), sz);
        if self.nrows() % sz.0 != 0 || self.ncols() % sz.1 != 0 {
            panic!("Matrix size should be a multiple of window size");
        }
        let pool_dims = (self.nrows() / sz.0, self.ncols() / sz.1);
        PatchIterator::<N, S, Dynamic, Dynamic> {
            source : &self,
            size : sz,
            curr_pos : (0, 0),
            _c_stride : self.nrows(),
            step_v,
            step_h,
            _row_wise : false,
            pool_dims
        }
    }
}

*/
