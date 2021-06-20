#[cfg(feature="mkl")]
mod fft1d;

#[cfg(feature="mkl")]
mod fft2d;

#[cfg(feature="mkl")]
pub use fft1d::*;

#[cfg(feature="mkl")]
pub use fft2d::*;
