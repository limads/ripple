/// General-purpose iterators over dynamic matrices.
pub mod iter;
use nalgebra::Scalar;
use std::ops::{Mul, AddAssign, Div, DivAssign, Add};
use num_traits::identities::{Zero, One};
use std::cmp::{Eq, PartialEq};
use num_traits::Float;
use std::any::Any;
use crate::signal::Signal;

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

/// Trait implemented by types which can be convolved with another instance
/// of themselves. Self must satisfy clone because the convolve(.) implementation
/// is provided by calling convolve_mut on a cloned instance.
pub trait Convolve
// where
//    Self : Clone
{

    fn convolve_mut(&self, filter : &Self, out : &mut Self);

    /*fn convolve(&self, filter : &Self) -> Self {
        let mut out = self.clone();
        self.convolve_mut(filter, &mut out);
        out
    }*/

}

impl<S> Convolve for [S]
where
    S : Scalar + Mul<Output = S> + Zero + AddAssign + Copy + Any
{

    fn convolve_mut(&self, filter : &Self, out : &mut Self) {

        #[cfg(feature="mkl")]
        if (&self[0] as &dyn Any).is::<f32>() {
            // Dispatch to MKL impl
        }

        #[cfg(feature="mkl")]
        if (&self[0] as &dyn Any).is::<f64>() {
            // Dispatch to MKL impl
        }

        baseline_convolution(self.as_ref(), filter.as_ref(), out.as_mut(), Extension::Ignore);
    }
}

impl<S> Convolve for Signal<S>
where
    S : Scalar + Mul<Output = S> + Zero + AddAssign + Copy + Any
{

    fn convolve_mut(&self, filter : &Self, out : &mut Self) {
        let input : &[S] = self.as_ref();
        let kernel : &[S] = filter.as_ref();
        let output : &mut [S] = out.as_mut();
        input.convolve_mut(kernel, output);
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
