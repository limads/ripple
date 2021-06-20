use crate::foreign::ipp::ippcore::ippMalloc;
use crate::foreign::ipp::ipps::*;
use std::ptr;
use std::convert::From;
use std::iter::FromIterator;
use crate::ipputils;
use crate::dwt::bank::*;
use std::convert::TryFrom;

#[derive(Clone, Copy, Debug)]
pub enum DWTSrcType {
    Float,
    Byte,
    Short,
    UnsignedShort
}

impl TryFrom<i32> for DWTSrcType {

    type Error = ();

    fn try_from(code : i32) -> Result<Self, ()> {
        if code == IppDataType_ipp32f {
            Ok(Self::Float)
        } else {
            if code == IppDataType_ipp8u {
                Ok(Self::Byte)
            } else {
                if code == IppDataType_ipp16s {
                    Ok(Self::Short)
                } else {
                    if code == IppDataType_ipp16u {
                        Ok(Self::UnsignedShort)
                    } else {
                        Err(())
                    }
                }
            }
        }
    }

}

impl Into<i32> for DWTSrcType {

    fn into(self) -> i32 {
        match self {
            Self::Float => IppDataType_ipp32f,
            Self::Byte => IppDataType_ipp8u,
            Self::Short => IppDataType_ipp16s,
            Self::UnsignedShort => IppDataType_ipp16u
        }
    }

}

/// Stores pairs of pointers to the forward and backward transforms
pub enum IppDWT {
    FloatToFloat(*mut IppsWTFwdState_32f, *mut IppsWTInvState_32f),
    ByteToFloat(*mut IppsWTFwdState_8u32f, *mut IppsWTInvState_32f8u),
    ShortToFloat(*mut IppsWTFwdState_16s32f, *mut IppsWTInvState_32f16s),
    UnsignedShortToFloat(*mut IppsWTFwdState_16u32f, *mut IppsWTInvState_32f16u)
}

impl IppDWT {

    // Returns IPP integer constant corresponding to the source type.
    pub fn src_type(&self) -> i32 {
        match self {
            IppDWT::FloatToFloat(_,_) => IppDataType_ipp32f,
            IppDWT::ByteToFloat(_,_) => IppDataType_ipp8u,
            IppDWT::ShortToFloat(_,_) => IppDataType_ipp16s,
            IppDWT::UnsignedShortToFloat(_,_) => IppDataType_ipp16u
        }
    }

}

// Calculate state structure size
unsafe fn calc_state_size(
    src_ty : DWTSrcType,
    len_low : i32,
    len_high : i32,
    offset_low : i32,
    offset_high : i32
) -> (i32, i32) {
    let dst_ty = IppDataType_ipp32f;
    let mut fwd_state_sz = 0;
    let status = ippsWTFwdGetSize(
        src_ty.into(),
        len_low,
        offset_low,
        len_high,
        offset_high,
        &mut fwd_state_sz as *mut _
    );
    ipputils::check_status("Get FWD size", status);

    //TODO verify if dst_ty is actualy the "destination" type from point of view of forward, not backward.
    let mut inv_state_sz = 0;
    let status = ippsWTInvGetSize(
        dst_ty,
        len_low,
        offset_low,
        len_high,
        offset_high,
        &mut inv_state_sz as *mut _
    );
    ipputils::check_status("Get BWD size", status);
    (fwd_state_sz, inv_state_sz)
}

pub unsafe fn build_dwt_state(src_ty : DWTSrcType, taps_low : &[f32], taps_high : &[f32]) -> IppDWT {
    let len_low = taps_low.len() as i32;
    let len_high = taps_high.len() as i32;
    let offset_low = 0;
    let offset_high = 0;

    // Calculate state structure size
    let (fwd_sz, inv_sz) = calc_state_size(
        src_ty,
        len_low,
        len_high,
        offset_low,
        offset_high
    );

    let mut ipp_dwt = IppDWT::FloatToFloat(ptr::null_mut(), ptr::null_mut());

    // Allocate state structure
    let (status_fwd, status_bwd) = match src_ty {
        DWTSrcType::Float => {
            let fwd_state : *mut IppsWTFwdState_32f = ippMalloc(fwd_sz) as *mut IppsWTFwdState_32f;
            let status_fwd = ippsWTFwdInit_32f(
                fwd_state,
                taps_low.as_ptr(),
                len_low,
                offset_low,
                taps_high.as_ptr(),
                len_high,
                offset_high
            );
            let inv_state : *mut IppsWTInvState_32f = ippMalloc(inv_sz) as *mut IppsWTInvState_32f;
            let status_bwd = ippsWTInvInit_32f(
                inv_state,
                taps_low.as_ptr(),
                len_low,
                offset_low,
                taps_high.as_ptr(),
                len_high,
                offset_high
            );
            ipp_dwt = IppDWT::FloatToFloat(fwd_state, inv_state);
            (status_fwd, status_bwd)
        },
        DWTSrcType::Byte => {
            unimplemented!()
        },
        DWTSrcType::Short => {
            unimplemented!()
        },
        DWTSrcType::UnsignedShort => {
            unimplemented!()
        }
    };

    ipputils::check_status("FWD init", status_fwd);
    ipputils::check_status("BWD init", status_bwd);

    // Solve wrapping problem

    ipp_dwt
}

/*unsafe fn build_filter_bwd(taps_low : &[f32], taps_high : &[f32]) -> *mut IppsWTInvState_32f {
    let len_low = taps_low.len() as i32;
    let len_high = taps_high.len() as i32;
    let offset_low = 0;
    let offset_high = 0;

    // Calculate state structure size
    // Allocate state structure

    inv_state
}*/

pub unsafe fn apply_custom_filter_fwd(
    state : &IppDWT,
    src : &[f32],
    dst_low : &mut [f32],
    dst_high : &mut [f32],
    filt_len : usize,
) {
    assert!(dst_low.len() == dst_high.len() && dst_low.len() == (src.len() / 2) );
    let status = match state {
        IppDWT::FloatToFloat(fwd_state, _) => {
            let status = ippsWTFwdSetDlyLine_32f(
                *fwd_state,
                &src[src.len() - filt_len / 2] as *const _,
                &src[src.len() - filt_len / 2] as *const _
            );
            ipputils::check_status("Set delay line FWD", status);
            ippsWTFwd_32f(
                src.as_ptr(),
                dst_low.as_mut_ptr(),
                dst_high.as_mut_ptr(),
                dst_low.len() as i32,
                *fwd_state
            )
        },
        _ => unimplemented!()
    };
    ipputils::check_status("Apply FWD", status);
}

pub unsafe fn apply_custom_filter_bwd(
    state : &IppDWT,
    src_low : &[f32],
    src_high : &[f32],
    dst : &mut [f32],
) {
    assert!(src_low.len() == src_high.len() && src_low.len() == (dst.len() / 2) );
    let status = match state {
        IppDWT::FloatToFloat(_, bwd_state) => {
            ippsWTInv_32f(
                src_low.as_ptr(),
                src_high.as_ptr(),
                src_low.len() as i32,
                dst.as_mut_ptr(),
                *bwd_state
            )
        },
        _ => unimplemented!()
    };
    ipputils::check_status("Apply BWD", status);
}

/* substitute for either 16s, 32f or 64f
For the 16s case:
Scale factor: Output is multiplied by 2^(-scale_factor). Applying this scale factor avoids saturation of integer
output values for restrictive data types such as the short (the user can then apply the factor to
the converted value in a type with higher range such as integer).
IppStatus ippsWTHaarFwd_16s_Sfs(const Ipp16s* pSrc, int len, Ipp16s* pDstLow, Ipp16s*
pDstHigh, int scaleFactor ); */
unsafe fn apply_haar_fwd(src : &[f32], dst_low : &mut [f32], dst_high : &mut [f32]) {
    let src_len = src.len() as i32;
    let status = ippsWTHaarFwd_32f(src.as_ptr(), src_len, dst_low.as_mut_ptr(), dst_high.as_mut_ptr());
    ipputils::check_status("Haar forward", status);
}

unsafe fn apply_haar_inv(src_low : &[f32], src_high : &[f32], dst : &mut [f32]) {
    let dst_len = dst.len() as i32;
    let status = ippsWTHaarInv_32f(src_low.as_ptr(), src_high.as_ptr(), dst.as_mut_ptr(), dst_len);
    ipputils::check_status("Haar backward", status);
}

/*fn apply_recursive(src : &[f32], dst : &mut [f32]) {
    // forward : apply direct recursively to the coarse output at each level, until a single coarse component is left.
    // backward : apply inverse recursively by adding the detail to the coarse data.
}*/

