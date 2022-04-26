#[cfg(feature="mkl")]
mod mkl;

#[cfg(feature="mkl")]
pub use mkl::fft1d::*;

#[cfg(feature="mkl")]
pub use mkl::fft2d::*;

#[cfg(not(feature="mkl"))]
mod rfft;

#[cfg(not(feature="mkl"))]
pub use rfft::*;


