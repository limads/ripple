#![doc(html_logo_url = "https://raw.githubusercontent.com/limads/ripple/master/assets/ripple-logo.png")]

pub(crate) mod foreign;

pub mod signal;

pub mod fft;

pub mod dwt;

pub(crate) fn panic_on_invalid_slices<A, B>(a : &[A], b : &[B]) {
    if a.len() == 0 {
        panic!("Input slice has zero lenght");
    }
    if b.len() == 0 {
        panic!("Output slice has zero lenght");
    }
    if a.len() != b.len() {
        panic!("Input and output slices differ in lenght ({} vs. {})", a.len(), b.len());
    }
}

#[cfg(feature="ipp")]
pub(crate) mod ipputils {

    use std::mem;
    use std::ptr;
    use std::ffi;
    use crate::foreign::ipp::ipps;
    use nalgebra::Scalar;
    use crate::foreign::ipp::ippcore::{self, ippMalloc};

    pub enum ScalarSlice<'a> {
        U8(&'a [u8]),
        I16(&'a [i16]),
        I32(&'a [i32]),
        F32(&'a [f32]),
        F64(&'a [f64])
    }

    impl<'a> From<&'a [u8]> for ScalarSlice<'a> {

        fn from(s : &'a [u8]) -> Self {
            ScalarSlice::U8(s)
        }

    }

    impl<'a> From<&'a [i16]> for ScalarSlice<'a> {

        fn from(s : &'a [i16]) -> Self {
            ScalarSlice::I16(s)
        }

    }

    impl<'a> From<&'a [i32]> for ScalarSlice<'a> {

        fn from(s : &'a [i32]) -> Self {
            ScalarSlice::I32(s)
        }

    }

    impl<'a> From<&'a [f32]> for ScalarSlice<'a> {

        fn from(s : &'a [f32]) -> Self {
            ScalarSlice::F32(s)
        }

    }

    impl<'a> From<&'a [f64]> for ScalarSlice<'a> {

        fn from(s : &'a [f64]) -> Self {
            ScalarSlice::F64(s)
        }

    }

    pub enum ScalarSliceMut<'a> {
        U8(&'a mut [u8]),
        I16(&'a mut [i16]),
        I32(&'a mut [i32]),
        F32(&'a mut [f32]),
        F64(&'a mut [f64])
    }

    impl<'a> From<&'a mut [u8]> for ScalarSliceMut<'a> {

        fn from(s : &'a mut [u8]) -> Self {
            ScalarSliceMut::U8(s)
        }

    }

    impl<'a> From<&'a mut [i16]> for ScalarSliceMut<'a> {

        fn from(s : &'a mut [i16]) -> Self {
            ScalarSliceMut::I16(s)
        }

    }

    impl<'a> From<&'a mut [i32]> for ScalarSliceMut<'a> {

        fn from(s : &'a mut [i32]) -> Self {
            ScalarSliceMut::I32(s)
        }

    }

    impl<'a> From<&'a mut [f32]> for ScalarSliceMut<'a> {

        fn from(s : &'a mut [f32]) -> Self {
            ScalarSliceMut::F32(s)
        }

    }

    impl<'a> From<&'a mut [f64]> for ScalarSliceMut<'a> {

        fn from(s : &'a mut [f64]) -> Self {
            ScalarSliceMut::F64(s)
        }

    }

    pub fn row_size_bytes<T>(ncol : usize) -> i32 {
        (ncol * mem::size_of::<T>()) as i32
    }

    pub fn check_status(action : &str, status : i32) {
        if status as u32 == ippcore::ippStsNoErr {
            return;
        }
        let err_msg : &'static str = match status {
            ippcore::ippStsNullPtrErr => "Null pointer",
            ippcore::ippStsNumChannelsErr => "Wrong number of channels",
            ippcore::ippStsAnchorErr => "Anchor error",
            ippcore::ippStsSizeErr => "Size error",
            ippcore::ippStsStepErr => "Step error",
            ippcore::ippStsContextMatchErr => "Context match error",
            ippcore::ippStsMemAllocErr => "Memory allocation error",
            ippcore::ippStsBadArgErr => "Bad argument",
            _ => "Unknown error"
        };
        panic!("IPPS Error\tAction: {}\tCode: {}\tMessage: {}", action, status, err_msg);
    }

    pub fn slice_size_bytes<T>(s : &[T]) -> i32 {
        (s.len() * mem::size_of::<T>()) as i32
    }

}
