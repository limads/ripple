use crate::foreign::ipp::ippi::*;
use crate::foreign::ipp::ippcore::{self, ippMalloc};
use crate::dwt::dwt1d;
use crate::ipputils::check_status;
use crate::ipputils::{self, row_size_bytes};
use nalgebra::DMatrix;
use crate::dwt::*;

pub struct IppDWT2D {
    pub buf_fwd : *mut u8,
    pub buf_bwd : *mut u8,
    pub spec_fwd : *mut IppiWTFwdSpec_32f_C1R,
    pub spec_bwd : *mut IppiWTInvSpec_32f_C1R
}

struct DWT2DSize {
    spec_fwd : i32,
    spec_bwd : i32,
    buf_fwd : i32,
    buf_bwd : i32
}

/*pub struct Wavelet2D<N> {
    levels : Vec<WaveletLevel<N>>
}

pub struct WaveletLevel<N> {
    src : Image<N>,
    dst : Image<N>
}*/

/// Returns required border size, as (nrow, ncol), when an image of
/// src_width x src_height is convolved with filters of len_low, len_high
/// wth anchor at filter indices anchor_low, anchor_high.
/// For a filter of size k, if anchor = 0, the full image should be padded with k-1 pixels
/// to the left, right, top and bottom portions. The left and top borders will have the same
/// size; the right and bottom borders will also have the same size. The function return the
/// value pair corresponding to those borders. For all anchor values, the allocated image size
/// will be the same (the left/top + bottom/right should sum to k-1), but the border size will
/// be different. For anchor = 0 left/top borders will be k-1 and the right/bottom borders
/// will be 0. For increasing anchor size, the top/left value will decrease and the right/bottom
/// borders will decrease.
fn border_size(
    src_width : usize,
    src_height : usize,
    filt_len_low : usize,
    filt_len_high : usize,
    anchor_low : usize,
    anchor_high : usize
) -> (usize, usize) {
    let left_border_low = filt_len_low - 1 - anchor_low;
    let left_border_high = filt_len_high - 1 - anchor_high;
    let right_border_low = anchor_low;
    let right_border_high = anchor_high;
    let left_top_border = left_border_low.max(left_border_high);
    let right_bottom_border = right_border_low.max(right_border_high);
    (left_top_border, right_bottom_border)
}

/// Returns extended required image size, as (nrow, ncol), when an image of
/// src_width x src_height is convolved with filters of len_low, len_high
/// wth anchor anchor_low, anchor_high.
fn extended_image_size(
    src_width : usize,
    src_height : usize,
    filt_len_low : usize,
    filt_len_high : usize,
    anchor_low : usize,
    anchor_high : usize
) -> (usize, usize) {
    // Required full image size:
    let (left_top_border, right_bottom_border) = border_size(
        src_width,
        src_height,
        filt_len_low,
        filt_len_high,
        anchor_low,
        anchor_high
    );
    let src_width_with_borders = src_width + left_top_border + right_bottom_border;
    let src_height_with_borders = src_height + left_top_border + right_bottom_border;
    (src_width_with_borders, src_height_with_borders)
}

unsafe fn calc_state_size(
    len_low : i32,
    len_high : i32,
    anchor_low : i32,
    anchor_high : i32
) -> DWT2DSize {
    let num_channels = 1;
    let mut spec_fwd = 0;
    let mut buf_fwd = 0;
    let get_sz_status = ippiWTFwdGetSize_32f(
        num_channels,
        len_low,
        anchor_low,
        len_high,
        anchor_high,
        &mut spec_fwd as *mut _,
        &mut buf_fwd as *mut _
    );
    check_status("Get fwd size", get_sz_status);

    let mut spec_bwd = 0;
    let mut buf_bwd = 0;
    let get_sz_status = ippiWTInvGetSize_32f(
        num_channels,
        len_low,
        anchor_low,
        len_high,
        anchor_high,
        &mut spec_bwd as *mut _,
        &mut buf_bwd as *mut _
    );
    check_status("Get bwd size", get_sz_status);
    DWT2DSize {
        spec_fwd,
        buf_fwd,
        spec_bwd,
        buf_bwd
    }
}

pub unsafe fn build_dwt2d_state(taps_low : &[f32], taps_high : &[f32]) -> IppDWT2D {
    assert!(taps_low.len() == 4 && taps_high.len() == 4);
    let len_low = taps_low.len() as i32;
    let len_high = taps_high.len() as i32;
    let anchor_low = taps_low.len() as i32 - 1; // 0
    let anchor_high = taps_high.len() as i32 - 1; // 0
    let sz = calc_state_size(len_low, len_high, anchor_low, anchor_high);

    let spec_fwd = ippMalloc(sz.spec_fwd) as *mut IppiWTFwdSpec_32f_C1R;
    let fwd_status = ippiWTFwdInit_32f_C1R(
        spec_fwd,
        taps_low.as_ptr(),
        len_low,
        anchor_low,
        taps_high.as_ptr(),
        len_high,
        anchor_high
    );
    check_status("init forward", fwd_status);
    let buf_fwd = ippMalloc(sz.buf_fwd) as *mut u8;

    let spec_bwd = ippMalloc(sz.spec_bwd) as *mut IppiWTInvSpec_32f_C1R;
    let bwd_status = ippiWTInvInit_32f_C1R(
        spec_bwd,
        taps_low.as_ptr(),
        len_low,
        anchor_low,
        taps_high.as_ptr(),
        len_high,
        anchor_high
    );
    check_status("init backward", bwd_status);
    let buf_bwd = ippMalloc(sz.buf_bwd) as *mut u8;
    IppDWT2D { spec_fwd, spec_bwd, buf_fwd, buf_bwd }
}

struct DWTParams {
    src_step_bytes : i32,
    dst_step_bytes : i32,
    dst_roi : IppiSize
}

impl DWTParams {

    fn define(src_len : usize, src_ncol : usize, filt_len : usize) -> Self {
        // src_ncol should be even.
        assert!(src_ncol % 2 == 0);

        // Assert image symmetry
        assert!(src_len / src_ncol == src_ncol);

        // The anchor represent the filter index which is positioned at the top-left image element.
        // This means all filter entries before it will be wrapped to the image end.
        // To avoid the user having to think about border expansion, we can set the anchor
        // to the maximum value (for 4-length filter, anchor=(3,3), which leaves the top-left
        // border as zero; but we "lie" to IPP that we have an image with size N-filt_len, so it will automatically
        // consider the remaining image entries as the required borders (lenght=filt_len-1) (at the cost of ignoring the
        // last image row).
        let anchor = filt_len - 1;
        let (tl_border, br_border) = border_size(src_ncol, src_ncol, filt_len, filt_len, anchor, anchor);
        assert!(tl_border == 0);
        assert!(br_border == filt_len - 1);

        let effective_ncol = src_ncol - filt_len;

        // Note we ignore the last image row, which is why we add a one
        let px_stride = effective_ncol + tl_border + br_border + 1;
        assert!(px_stride == src_ncol);

        // Total padding applied to the top-left of the original buffer slice: Since the top and
        // left borders are the same, count (ncol + left) border (top) times. Use roi_slice
        // instead of the actual data slice for non-zero TL border.
        // let tl_linear_pad = tl_border*px_stride + tl_border;
        // assert!(tl_linear_pad == 0 );
        // let roi_slice = &src[tl_linear_pad..];

        let src_step_bytes = ipputils::row_size_bytes::<f32>(px_stride);
        let dst_step_bytes = ipputils::row_size_bytes::<f32>(effective_ncol / 2);
        let dst_roi = IppiSize{ width : (effective_ncol / 2) as i32, height : (effective_ncol / 2) as i32 };
        DWTParams { src_step_bytes, dst_step_bytes, dst_roi }
    }
}

/// Note: src slice should point to inner image region, to account for DWT borders.
/// approx/detail_x/detail_y/detail_xy: Destination slices of size (nrow / 2, ncol / 2) each
/// src: Padded Source slice of size (tl+nrow,tl+ncol)
/// ncol: ROI number of columns (assumed to be equal to number of rows)
pub unsafe fn apply_forward(
    fwd_spec : *const IppiWTFwdSpec_32f_C1R,
    fwd_buf : *mut u8,
    src : &[f32],
    src_ncol : usize,
    filt_len : usize,
    approx : &mut [f32],
    detail_x : &mut [f32],
    detail_y : &mut [f32],
    detail_xy : &mut [f32]
) {
    let params = DWTParams::define(src.len(), src_ncol, filt_len);
    let fwd_status = ippiWTFwd_32f_C1R (
        src.as_ptr(),
        params.src_step_bytes,
        approx.as_mut_ptr(),
        params.dst_step_bytes,
        detail_x.as_mut_ptr(),
        params.dst_step_bytes,
        detail_y.as_mut_ptr(),
        params.dst_step_bytes,
        detail_xy.as_mut_ptr(),
        params.dst_step_bytes,
        params.dst_roi,
        fwd_spec,
        fwd_buf
    );
    check_status("apply forward", fwd_status);
}

pub unsafe fn apply_backward(
    bwd_spec : *const IppiWTInvSpec_32f_C1R,
    bwd_buf : *mut u8,
    src : &mut [f32],
    src_ncol : usize,
    filt_len : usize,
    approx : &[f32],
    detail_x : &[f32],
    detail_y : &[f32],
    detail_xy : &[f32]
) {
    let params = DWTParams::define(src.len(), src_ncol, filt_len);
    let bwd_status = ippiWTInv_32f_C1R(
        approx.as_ptr(),
        params.dst_step_bytes,
        detail_x.as_ptr(),
        params.dst_step_bytes,
        detail_y.as_ptr(),
        params.dst_step_bytes,
        detail_xy.as_ptr(),
        params.dst_step_bytes,
        params.dst_roi,
        src.as_mut_ptr(),
        params.src_step_bytes,
        bwd_spec,
        bwd_buf
    );
    check_status("apply backward", bwd_status);
}

// cargo test --all-features test_extended_image_size -- --nocapture
#[test]
fn test_extended_image_size() {
    /// Anchor is an index of the filter that will correspond to the top-left-most image pixel.
    // Therefore, it should be smaller than or equal to filter len - 1.
    // Here, we verify by how much a 64x64 image should be extended for all possible
    // anchors of a 4x4 filter.
    let anchors : [usize; 4] = [0, 1, 2, 3];
    let filt_len_low = 4;
    let filt_len_high = 4;
    for anchor in &anchors {
        let (left_top, right_bottom) = border_size(64, 64, filt_len_low, filt_len_high, *anchor, *anchor);
        println!("Left and top borders = {:?}; Right and bottom borders = {:?}", left_top, right_bottom);
        let extended_sz = extended_image_size(64, 64, filt_len_low, filt_len_high, *anchor, *anchor);
        println!("Anchor = {:?}; Extended size = {:?}", (anchor, anchor), extended_sz);
    }
}

fn mat_from_slice(s : &[f32], ncols : usize) -> DMatrix<f32> {
    DMatrix::from_iterator(ncols, s.len() / ncols, s.iter().cloned()).transpose()
}

// cargo test --all-features image_dwt -- --nocapture
#[test]
fn image_dwt() {

    use crate::dwt::bank;
    use crate::signal::gen;
    use std::iter::FromIterator;

    // For anchor = 0, A 4x4 image should be padded to a 7x7 image (filter len-1), which has 49 entries.
    // If we assume anchor=0, the padding is in the first 3 rows and columns.
    //                  | Effective image starts here
    //                  V
    /*let img : [f32; 49] = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // <-- Effective image starts here
        0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
    ];*/

    let img_len = 64;
    let filt_len = 4;
    let half_len = img_len / 2;
    // Create step image
    let mut img = gen::flat(img_len);
    for i in 0..(img_len-1) {
        if i <= (half_len-1) {
            img.extend(gen::flat(img_len));
        } else {
            img.extend(gen::step(img_len));
        }
    }
    println!("{}", img.len());

    let dst_len = (half_len - filt_len / 2) as usize;
    // println!("dst len = {}", dst_len);

    let mut approx = Vec::from_iter((0..dst_len.pow(2u32)).map(|_| 0.0 ));
    let mut detail_x = approx.clone();
    let mut detail_y = approx.clone();
    let mut detail_xy = approx.clone();
    println!("Building filters");

    unsafe {
        let dwt2d = build_dwt2d_state(&bank::DAUBECHIES[..], &bank::advance_level(&bank::DAUBECHIES)[..]);
        apply_forward(
            dwt2d.spec_fwd,
            dwt2d.buf_fwd,
            img.as_ref(),
            img_len,
            filt_len,
            &mut approx[..],
            &mut detail_x[..],
            &mut detail_y[..],
            &mut detail_xy[..]
        );
    }

    let dst_offset = dst_len as usize;
    let dst_ncols = dst_len as usize;
    // let approx_mat = mat_from_slice(&approx[..], dst_len);
    // println!("{:?}", approx_mat.shape());
    println!("Approx = {:.4}", mat_from_slice(&approx[..], dst_len));
    println!("Detail X = {:.4}", mat_from_slice(&detail_x[..], dst_len));
    println!("Detail Y = {:.4}", mat_from_slice(&detail_y[..], dst_len));
    println!("Detail XY = {:.4}", mat_from_slice(&detail_xy[..], dst_len));

    /* Approx = [0.0, 0.0, 0.40400636, 1.291266]
    Detail X = [0.0, 0.0, -0.10825317, -0.34599364]
    Detail Y = [0.0, 0.0, -0.10825317, 0.40400636]
    Detail XY = [0.0, 0.0, 0.02900635, -0.10825318] */
}
