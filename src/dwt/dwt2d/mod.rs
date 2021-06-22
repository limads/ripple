use nalgebra::*;
// use iter::*;
use nalgebra::storage::*;
// use super::gsl::*;
// use crate::series::*;
// use crate::image::*;
// use super::dwt1d::gsl::*;
use std::ops::{Mul, MulAssign, Index, };
use crate::dwt::*;
use std::iter::FromIterator;

pub mod ipp;

// pub mod iter;

// use iter::DWTIteratorBase;

/// Two-dimensional wavelet decomposition
pub struct Wavelet2D {
    states : Vec<ipp::IppDWT2D>,
    coefs : Vec<([f32; 4], [f32; 4])>,
    basis : Basis,
    img_side : usize
}

impl Wavelet2D {

    pub fn new(basis : Basis, img_side : usize, n_levels : usize) -> Self {
        let coefs = basis.taps(n_levels);
        let mut states = Vec::new();
        for i in 0..n_levels {
            let state = unsafe {
                ipp::build_dwt2d_state(&coefs[i].0[..], &coefs[i].1[..])
            };
            states.push(state);
        }
        Self { states, coefs, basis, img_side }
    }

    pub fn forward_mut(&self, src : &impl AsRef<[f32]>, dst : &mut Pyramid<f32>) {
        assert!(self.states.len() == dst.levels.len());
        unsafe {
            for i in 0..self.states.len() {
                let (coarse, detail_x, detail_y, detail_xy) = dst.full_level_mut(i);
                ipp::apply_forward(
                    self.states[i].spec_fwd,
                    self.states[i].buf_fwd,
                    src.as_ref(),
                    self.img_side,
                    self.basis.len(),
                    coarse,
                    detail_x,
                    detail_y,
                    detail_xy
                );
            }
        }
    }

    pub fn backward_mut(&self, src : &Pyramid<f32>, dst : &mut AsMut<[f32]>) {
        assert!(self.states.len() == src.levels.len());
        unsafe {
            for i in 0..self.states.len() {
                let (coarse, detail_x, detail_y, detail_xy) = src.full_level(i);
                ipp::apply_backward(
                    self.states[i].spec_bwd,
                    self.states[i].buf_bwd,
                    dst.as_mut(),
                    self.img_side,
                    self.basis.len(),
                    &coarse[..],
                    &detail_x[..],
                    &detail_y[..],
                    &detail_xy[..]
                );
            }
        }
    }

}

pub struct PyramidLevel<N> {
    detail_x : Vec<N>,
    detail_y : Vec<N>,
    detail_xy : Vec<N>,
    coarse : Vec<N>,
    side_len : usize
}

impl<N> PyramidLevel<N>
where
    N : From<f32> + Scalar + Clone
{

    pub fn new(side_len : usize, filt_len : usize) -> Self {
        let half_len = side_len / 2;
        let dst_len = (half_len - filt_len / 2) as usize;
        let mut coarse = Vec::from_iter((0..dst_len.pow(2u32)).map(|_| N::from(0.0) ));
        let mut detail_x = coarse.clone();
        let mut detail_y = coarse.clone();
        let mut detail_xy = coarse.clone();
        Self {
            detail_x,
            detail_y,
            detail_xy,
            coarse,
            side_len
        }
    }

}

pub struct Pyramid<N>
where
    N : From<f32> + Scalar + Clone
{
    levels : Vec<PyramidLevel<N>>
}

impl<N> Pyramid<N>
where
    N : From<f32> + Scalar + Clone
{

    pub fn horizontal_mut(&mut self, level : usize) -> &mut [N] {
        &mut self.levels[level].detail_x[..]
    }

    pub fn vertical_mut(&mut self, level : usize) -> &mut [N] {
        &mut self.levels[level].detail_y[..]
    }

    pub fn diagonal_mut(&mut self, level : usize) -> &mut [N] {
        &mut self.levels[level].detail_xy[..]
    }

    pub fn coarse_mut(&mut self, level : usize) -> &mut [N] {
        &mut self.levels[level].coarse[..]
    }

    fn full_level_mut(&mut self, level : usize) -> (&mut [N], &mut [N], &mut [N], &mut [N]) {
        let level = &mut self.levels[level];
        (
            &mut level.coarse[..],
            &mut level.detail_x[..],
            &mut level.detail_y[..],
            &mut level.detail_xy[..]
        )
    }

    fn full_level(&self, level : usize) -> (&[N], &[N], &[N], &[N]) {
        let level = &self.levels[level];
        (
            &level.coarse[..],
            &level.detail_x[..],
            &level.detail_y[..],
            &level.detail_xy[..]
        )
    }

}

/// Output of a wavelet decomposition. Imgage pyramids are indexed by a (scale, x, y) triplet.
/*#[derive(Clone, Debug)]
pub struct ImagePyramid<N>
where
    N : Scalar + Copy
{
    pyr : Image<N>
}

impl<N> ImagePyramid<N>
where
    N : Scalar + Copy
{

    pub fn len(&self) -> usize {
        self.pyr.len()
    }
}

impl<N> ImagePyramid<N>
where
    N : Scalar + Copy
{

    pub fn new_constant(n : usize, value : N) -> Self {
        Self{ pyr : Image::new_constant(n, n, value) }
    }

    pub fn levels<'a>(&'a self) -> impl Iterator<Item=ImageLevel<'a, N>> {
        DWTIteratorBase::<&'a ImagePyramid<N>>::new_ref(&self)
    }

    /*pub fn levels_mut<'a>(&'a mut self) -> impl Iterator<Item=DMatrixSliceMut<'a, f64>> {
        DWTIteratorBase::<&'a mut DMatrix<f64>>::new_mut(&mut self.pyr)
    }*/
}

impl<N> AsRef<[N]> for ImagePyramid<N>
where
    N : Scalar + Copy
{
    fn as_ref(&self) -> &[N] {
        self.pyr.as_slice()
    }
}

impl<N> AsMut<[N]> for ImagePyramid<N>
where
    N : Scalar + Copy
{
    fn as_mut(&mut self) -> &mut [N] {
        self.pyr.as_mut_slice()
    }
}

pub struct ImageLevel<'a, N>
where
    N : Scalar + Copy
{
    win : Window<'a, N>
}

impl<'a, N> ImageLevel<'a, N>
where
    N : Scalar + Copy + Mul<Output=N> + MulAssign
{
    pub fn windows(&self, sz : (usize, usize)) -> impl Iterator<Item=Window<'a, N>> {
        self.win.clone().windows(sz)
    }

}

impl<'a, N> Index<(usize, usize)> for ImageLevel<'a, N>
where
    N : Scalar + Copy
{

    type Output = N;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.win[index]
    }
}

impl<'a, N> From<Window<'a, N>> for ImageLevel<'a, N>
where
    N : Scalar + Copy
{

    fn from(win : Window<'a, N>) -> Self {
        Self{ win }
    }
}

pub struct ImageLevelMut<'a, N>
where
    N : Scalar + Copy
{
    win : WindowMut<'a, N>
}

impl<'a, N> From<WindowMut<'a, N>> for ImageLevelMut<'a, N>
where
    N : Scalar + Copy
{

    fn from(win : WindowMut<'a, N>) -> Self {
        Self{ win }
    }
}

impl<N> AsRef<Image<N>> for ImagePyramid<N>
where
    N : Scalar + Copy
{
    fn as_ref(&self) -> &Image<N> {
        &self.pyr
    }
}

impl<N> AsMut<Image<N>> for ImagePyramid<N>
where
    N : Scalar + Copy
{
    fn as_mut(&mut self) -> &mut Image<N> {
        &mut self.pyr
    }
}

/*impl<N> From<DMatrix<N>> for ImagePyramid<N>
where
    N : Scalar
{
    fn from(s : DMatrix<N>) -> Self {
        Self{ pyr : s }
    }
}

impl<N> AsRef<DMatrix<N>> for ImagePyramid<N>
where
    N : Scalar
{
    fn as_ref(&self) -> &DMatrix<N> {
        &self.pyr
    }
}*/

impl<N> From<Vec<N>> for Pyramid<N>
where
    N : Scalar
{
    fn from(s : Vec<N>) -> Self {
        Self{ buf : DVector::from_vec(s) }
    }
}*/

impl Wavelet2D {

    /*pub fn new(basis : Basis, sz : usize) -> Result<Self, &'static str> {
        Ok(Self { plan : DWTPlan::new(basis, (sz, sz) )? })
    }

    pub fn forward_mut(&self, src : &Image<f64>, dst : &mut ImagePyramid<f64>) {
        self.plan.apply_forward(src.as_ref(), dst.as_mut())
            .map_err(|e| panic!("{}", e) );
    }

    pub fn forward(&self, src : &Image<f64>) -> ImagePyramid<f64> {
        let (nrows, ncols) = self.plan.shape();
        let mut dst = ImagePyramid::new_constant(nrows, 0.0);
        self.forward_mut(src, &mut dst);
        dst
    }

    pub fn forward_inplace(&self, mut buffer : Image<f64>) -> ImagePyramid<f64> {
        if let Err(e) = self.plan.forward_inplace(buffer.as_mut()) {
            panic!("{}", e);
        }
        ImagePyramid { pyr : buffer }
    }

    pub fn backward_mut(&self, src : &ImagePyramid<f64>, dst : &mut Image<f64>) {
        self.plan.apply_backward(src.as_ref(), dst.as_mut())
            .map_err(|e| panic!("{}", e) );
    }

    pub fn backward_inplace(&self, mut buffer : ImagePyramid<f64>) -> Image<f64> {
        if let Err(e) = self.plan.backward_inplace(buffer.as_mut()) {
            panic!("{}", e);
        }
        buffer.pyr
    }

    pub fn backward(&self, src : &ImagePyramid<f64>) -> Image<f64> {
        let (nrows, ncols) = self.plan.shape();
        let mut dst = Image::new_constant(nrows, ncols, 0.0);
        self.backward_mut(src, &mut dst);
        dst
    }*/
}

/*impl Forward<Image<f64>> for Wavelet2D {

    type Output = Image<f64>;

    fn forward_mut(&self, src : &Image<f64>, dst : &mut Self::Output) {
        self.plan.apply_forward(src.as_ref(), dst.as_mut())
            .map_err(|e| panic!("{}", e) );
    }

    fn forward(&self, src : &Image<f64>) -> Self::Output {
        let (nrows, ncols) = self.plan.shape();
        let mut dst = Image::new_constant(nrows, ncols, 0.0);
        self.forward_mut(src, &mut dst);
        dst
    }
}

impl Backward<Image<f64>> for Wavelet2D {

    type Output = Image<f64>;

    fn backward_mut(&self, src : &Image<f64>, dst : &mut Self::Output) {
        self.plan.apply_backward(src.as_ref(), dst.as_mut())
            .map_err(|e| panic!("{}", e) );
    }

    fn backward(&self, src : &Image<f64>) -> Self::Output {
        let (nrows, ncols) = self.plan.shape();
        let mut dst = Image::new_constant(nrows, ncols, 0.0);
        self.backward_mut(src, &mut dst);
        dst
    }
}*/

/*
If the interest is exclusively on image reconstruction from low spatial frequencies,
see opencv::imgproc::{pyr_down, pyr_up} as an alternative. If the interest is on the coefficients
themselves (e.g. keypoint extraction) then the DWT is the way to go.

*/
