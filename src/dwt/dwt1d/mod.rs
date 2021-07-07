use nalgebra::*;
use iter::*;
use nalgebra::storage::*;
use crate::signal::*;
use std::cell::RefCell;
use std::iter::FromIterator;
use crate::dwt::*;
use std::ops::Range;
use std::fmt::Debug;

// Utilities for iterating over the levels of a wavelet transform.
// pub mod iter;

mod ipp;

// pub mod iter;
// use iter::*;

/// Wavelet transform.
pub struct Wavelet {
    states : Vec<ipp::IppDWT>,
    /*coefs : Vec<([f32; 4], [f32; 4])>,*/
    basis : Basis,
    sz : usize,
    bwd_cascade : RefCell<Cascade<f32>>
}

unsafe impl Send for Wavelet { }

unsafe impl Sync for Wavelet { }

impl Wavelet {

    pub fn new(basis : Basis, mut sz : usize, n_levels : usize) -> Result<Self, &'static str> {
        assert!(n_levels >= 1 && n_levels <= basis.len());
        let (low, high) = basis.coefficients();
        let mut states = Vec::new();
        for i in 0..n_levels {
            let state = unsafe {
                ipp::build_dwt_state(ipp::DWTSrcType::Float, &low[..], &high[..])
            };
            states.push(state);
        }
        let bwd_cascade = RefCell::new(Cascade::new(basis, sz, Some(states.len())));
        Ok(Self { states, /*coefs,*/ basis, sz, bwd_cascade })
    }

    pub fn empty_cascade(&self) -> Cascade<f32> {
        Cascade::new(self.basis, self.sz, Some(self.states.len()))
    }

    pub fn forward_mut(&self, src : &impl AsRef<[f32]>, dst : &mut Cascade<f32>) {
        assert!(self.states.len() == dst.levels.len());
        unsafe {
            for i in 0..self.states.len() {
                if i == 0 {
                    let curr_lvl = &mut dst.levels[i];

                    // println!("src len = {}; dst len = {}", src.as_ref().len(), curr_lvl.detail.len());
                    super::verify_dwt1d_dimensions(src.as_ref(), &curr_lvl.coarse[..], &curr_lvl.detail[..]);
                    ipp::apply_custom_filter_fwd(
                        &self.states[i],
                        src.as_ref(),
                        &mut curr_lvl.coarse[..],
                        &mut curr_lvl.detail[..],
                        self.basis.len()
                    );
                } else {
                    let (prev_lvl, curr_lvl) = dst.level_pair_mut(i);

                    // println!("src len = {}; dst len = {}", src.as_ref().len(), curr_lvl.detail.len());
                    super::verify_dwt1d_dimensions(&prev_lvl.coarse[..], &curr_lvl.coarse[..], &curr_lvl.detail[..]);
                    ipp::apply_custom_filter_fwd(
                        &self.states[i],
                        &prev_lvl.coarse[..],
                        &mut curr_lvl.coarse[..],
                        &mut curr_lvl.detail[..],
                        self.basis.len()
                    );
                }
            }
        }
    }

    /*pub fn forward_inplace(&self, mut buffer : Signal<f64>) -> Pyramid<f64> {
        if let Err(e) = self.plan.forward_inplace(buffer.as_mut()) {
            panic!("{}", e);
        }
        Pyramid { buf : buffer.buf }
    }

    pub fn forward(&self, src : &Signal<f64>) -> Pyramid<f64> {
        let mut dst = Pyramid::new_constant(self.plan.shape().0, 0.0);
        self.forward_mut(src, &mut dst);
        dst
    }*/

    pub fn backward_mut(&self, src : &Cascade<f32>, dst : &mut impl AsMut<[f32]>) {
        assert!(self.states.len() == src.levels.len());
        let mut bwd_cascade = self.bwd_cascade.borrow_mut();
        unsafe {
            for i in (0..self.states.len()).rev().skip(1) {
                if i == 0 {
                    let curr_lvl = &bwd_cascade.levels[0];
                    ipp::apply_custom_filter_bwd(
                        &self.states[i],
                        curr_lvl.coarse.as_ref(),
                        curr_lvl.detail.as_ref(),
                        dst.as_mut()
                    );
                } else {
                    let prev_lvl = &src.levels[i];
                    let curr_lvl = &mut bwd_cascade.levels[i];
                    ipp::apply_custom_filter_bwd(
                        &self.states[i],
                        prev_lvl.coarse.as_ref(),
                        prev_lvl.detail.as_ref(),
                        curr_lvl.coarse.as_mut()
                    );
                }
            }
        }
    }

    /*pub fn backward_inplace(&self, mut buffer : Pyramid<f64>) -> Signal<f64> {
        if let Err(e) = self.plan.backward_inplace(buffer.as_mut()) {
            panic!("{}", e);
        }
        Signal { buf : buffer.buf }
    }

    pub fn backward(&self, src : &Pyramid<f64>) -> Signal<f64> {
        let mut dst = Signal::new_constant(self.plan.shape().0, 0.0);
        self.backward_mut(src, &mut dst);
        dst
    }*/
}

/*pub struct Scale<'a> {
    vals : DVectorSliceMut<'a, f64>
}

impl Iterator<Item=Scale<'a>> for ScaleIter {

    fn next(&mut self) -> Option<Scale<'a>> {
        if self.curr_lvl == self.max_lvl + 1 {
            return None;
        }
        self.curr_lvl += 1;
        Some(())
    }
}*/
/*struct ScaleMut<'a, N> {
    full : &'a mut Pyramid<N>
    curr : usize
}*/

#[derive(Clone, Debug)]
pub struct CascadeLevel<N>
    where N : Scalar
{
    coarse : Vec<N>,
    detail : Vec<N>
}

impl<N> CascadeLevel<N>
where
    N : From<f32> + Scalar
{

    pub fn new(len : usize, filt_len : usize) -> Self {
        // let half_len = len / 2;
        let mut coarse = Vec::from_iter((0..len).map(|_| N::from(0.0) ));
        let mut detail = coarse.clone();
        Self {
            coarse,
            detail
        }
    }

    pub fn coarse(&self) -> &[N] {
        &self.coarse[..]
    }

    pub fn detail(&self) -> &[N] {
        &self.detail[..]
    }
}

/// Output of a one-dimensional wavelet transform.
#[derive(Clone, Debug)]
pub struct Cascade<N>
where
    N : Scalar + Debug
{
    levels : Vec<CascadeLevel<N>>
}

impl<N> Cascade<N>
where
    N : Scalar + Copy + From<f32>
{

    pub fn level_pair_mut(&mut self, last : usize) -> (&mut CascadeLevel<N>, &mut CascadeLevel<N>) {
        index_and_prev_mut(&mut self.levels[..], last)
    }

    pub fn new(basis : Basis, len : usize, n_levels : Option<usize>) -> Self {
        let n_levels = n_levels.unwrap_or(dwt_max_levels(len));
        let filt_len = basis.len();
        let levels = (0..n_levels)
            .map(|lvl| CascadeLevel::new(len / (2usize).pow(lvl as u32 + 1), filt_len) )
            .collect();
        Self { levels }

        /*let low = Vec::from_iter((0..n_samples).map(|_| N::from(0.0 as f32) ));
        let high = low.clone();
        let n_levels = n_levels.unwrap_or(dwt_max_levels(n_samples));
        let levels = 0..n_levels;*/
        // Self{ low, high, levels }
    }

    pub fn levels(&self) -> impl Iterator<Item=&CascadeLevel<N>> {
        self.levels.iter()
    }

    /*pub fn detail<'a>(&'a self, lvl : usize) -> &'a [N] {
        let n = self.low.len();
        &self.high[dwt_level_index(n, lvl)]
    }

    pub fn coarse<'a>(&'a self, lvl : usize) -> &'a [N] {
        let n = self.low.len();
        &self.low[dwt_level_index(n, lvl)]
    }

    pub fn detail_mut<'a>(&'a mut self, lvl : usize) -> &'a mut [N] {
        let n = self.low.len();
        &mut self.high[dwt_level_index(n, lvl)]
    }

    pub fn coarse_mut<'a>(&'a mut self, lvl : usize) -> &'a mut [N] {
        let n = self.low.len();
        &mut self.low[dwt_level_index(n, lvl)]
    }

    fn coarse_detail_mut<'a>(&'a mut self, lvl : usize) -> (&'a mut [N], &'a mut [N]) {
        let n = self.low.len();
        (&mut self.low[dwt_level_index(n, lvl)], &mut self.high[dwt_level_index(n, lvl)])
    }

    fn prev_coarse_detail_mut<'a>(&'a mut self, lvl : usize) -> (&'a mut [N], &'a mut [N], &'a mut [N]) {
        assert!(lvl >= 1);
        /*let n = self.low.len();
        let low_prev_ix = dwt_level_index(n, lvl-1);
        let low_curr_ix = dwt_level_index(n, lvl);
        (&mut self.low[], &mut self.low[], &mut self.high[dwt_level_index(n, lvl)])*/
        unimplemented!()
    }

    fn coarse_detail<'a>(&'a self, lvl : usize) -> (&'a [N], &'a [N]) {
        let n = self.low.len();
        (&self.low[dwt_level_index(n, lvl)], &self.high[dwt_level_index(n, lvl)])
    }

    pub fn detail_iter<'a>(&'a self) -> impl Iterator<Item=&'a [N]> {
        self.levels.clone().map(move |lvl| &self.high[dwt_level_index(self.low.len(), lvl)] )
    }

    pub fn coarse_iter<'a>(&'a self) -> impl Iterator<Item=&'a [N]> {
        self.levels.clone().map(move |lvl| &self.low[dwt_level_index(self.low.len(), lvl)] )
    }

    // pub fn detail_levels_mut(&mut self) -> impl Iterator<Item=&mut [N]> {
    //    self.levels.clone().map(move |lvl| &mut self.high[dwt_level_index(self.low.len(), lvl)] )
    // }

    /*pub fn coarse_levels_mut<'a>(&'a mut self) -> impl Iterator<Item=&'a mut [N]> {
        self.levels.clone().map(move |lvl| &mut self.low[dwt_level_index(self.low.len(), lvl)] )
    }*/

    /*pub fn levels<'a>(&'a self) -> impl Iterator<Item=DVectorSlice<'a, N>> {
        DWTIteratorBase::<&'a Pyramid<N>>::new_ref(&self)
    }*/

    /// Shrinks all coefficients at this level by the given scalar value
    pub fn shrink(&mut self, factor : f64) {
        unimplemented!()
    }

    /// Sets this factor to zero
    pub fn truncate(&mut self) {
        unimplemented!()
    }

    pub fn len(&self) -> usize {
        self.buf.nrows()
    }

    pub fn levels_mut<'a>(&'a mut self) -> impl Iterator<Item=DVectorSliceMut<'a, f64>> {
        DWTIteratorBase::<&'a mut DVector<f64>>::new_mut(&mut self.buf)
    }*/
}

// cargo test --all-features dwt_1d -- --nocapture
#[test]
fn dwt_1d() {

    use crate::signal::gen;

    // let mut low = gen::flat(32);
    // let mut high = low.clone();

    // Flat: Both low and high are also flat;
    // Ramp: Low is flat at sqrt(2); High is 0.0;
    // Step: Low is 0.0 before 16; sqrt(2) after 16; High is 0.0 before 16; 0.48 at 16; Then 0.0 again;

    let names = ["flat", "ramp", "step"];
    let mut data = [gen::flat(64), gen::ramp(64), gen::step(64)];
    let wav = Wavelet::new(Basis::Daubechies(4, false), 64, 1).unwrap();
    let mut casc = wav.empty_cascade();
    unsafe {
        // let wav = bank::DAUB4_LOW;
        // let state = build_dwt_state(DWTSrcType::Float, &wav[..], &bank::advance_level(&wav)[..]);
        // let state_bwd : *mut IppsWTInvState_32f = build_filter_bwd(&wav[..], &bank::advance_level(&wav)[..]);
        for (name, data) in names.iter().zip(data.iter_mut()) {

            // apply_custom_filter_fwd(&state, data.as_ref(), low.as_mut(), high.as_mut(), wav.len());
            // apply_custom_filter_bwd(&state, low.as_ref(), high.as_ref(), data.as_mut());
            wav.forward_mut(&data, &mut casc);

            println!("{}", name);
            println!("Low: {:?}", &casc.coarse(0)[4..28]);
            println!("High: {:?}", &casc.detail(0)[4..28]);

            // println!("Inverse: {:?}", &data.as_slice()[4..60]);
        }
    }

}

/*impl<N> AsRef<[N]> for Pyramid<N>
where
    N : Scalar
{
    fn as_ref(&self) -> &[N] {
        self.buf.data.as_slice()
    }
}

impl<N> AsMut<[N]> for Pyramid<N>
where
    N : Scalar
{
    fn as_mut(&mut self) -> &mut [N] {
        self.buf.data.as_mut_slice()
    }
}

impl<N> AsRef<DVector<N>> for Pyramid<N>
where
    N : Scalar
{
    fn as_ref(&self) -> &DVector<N> {
        &self.buf
    }
}

impl<N> From<DVector<N>> for Pyramid<N>
where
    N : Scalar
{
    fn from(s : DVector<N>) -> Self {
        Self{ buf : s }
    }
}

impl<N> From<Vec<N>> for Pyramid<N>
where
    N : Scalar
{
    fn from(s : Vec<N>) -> Self {
        Self{ buf : DVector::from_vec(s) }
    }
}*/

/*impl Forward<Signal<f64>> for Wavelet {

    type Output = Signal<f64>;

    fn forward_mut(&self, src : &Signal<f64>, dst : &mut Self::Output) {
        self.plan.apply_forward(src.as_ref(), dst.as_mut())
            .map_err(|e| panic!("{}", e) );
    }

    fn forward(&self, src : &Signal<f64>) -> Self::Output {
        let mut dst = Signal::new_constant(self.plan.shape().0, 0.0);
        self.forward_mut(src, &mut dst);
        dst
    }
}

impl Backward<Signal<f64>> for Wavelet {

    type Output = Signal<f64>;

    fn backward_mut(&self, src : &Signal<f64>, dst : &mut Self::Output) {
        self.plan.apply_backward(src.as_ref(), dst.as_mut())
            .map_err(|e| panic!("{}", e) );
    }

    fn backward(&self, src : &Signal<f64>) -> Self::Output {
        let mut dst = Signal::new_constant(self.plan.shape().0, 0.0);
        self.backward_mut(src, &mut dst);
        dst
    }

}*/

/*#[derive(Clone)]
pub struct DWT {
    plan : DWTPlan1D,
    domain : Option<DVector<f64>>,
    // back_call : bool
}

impl DWT {

    /// Builds a new 1D discrete wavelet transform from the zeroed-out
    /// buffer with the given number of rows.
    pub fn new_empty(nrows : usize) -> Self {
        Self::new(DVector::zeros(nrows))
    }

    /// Builds a new 1D discrete wavelet transform from the informed source.
    pub fn new<S>(s : S) -> Self
        where S : Into<DVector<f64>>
    {
        let mut vs = s.into();
        let mut plan = DWTPlan1D::new(gsl::Basis::Daubechies(6, true), vs.nrows()).unwrap();
        let _ = plan.forward(&vs);
        let domain = Some(vs);
        //let back_call = true;
        //Self{ plan, domain, back_call }
        Self{ plan, domain }
    }

    pub fn iter_levels<'a>(&'a self) -> DWTIterator1D<'a> {
        self.plan.iter_levels()
    }

    pub fn iter_levels_mut<'a>(&'a mut self) -> DWTIteratorMut1D<'a> {
        self.plan.iter_levels_mut()
    }

}

impl<'a, S> Forward<'a, Matrix<f64, Dynamic, U1, S>, DVector<f64>> for DWT
where S : ContiguousStorage<f64, Dynamic, U1>
{

    fn forward_from(&'a mut self, s : &Matrix<f64, Dynamic, U1, S>) -> &'a DVector<f64> {
        let _ = self.plan.forward(&s);
        &self.plan.buf
        //self.back_call = false;
        //&(*ans)
    }

    /*fn partial_backward<S>(&'a mut self, n : usize) -> DVectorSlice<'a, f64> {
        //let mut dst = self.plan.take().unwrap();
        //plan.backward_to(&mut dst);
        //self.dst = Some(dst);
        //dst.as_ref().into()
        unimplemented!()
    }*/

    fn coefficients(&'a self) ->  &'a DVector<f64> {
        &self.plan.buf
    }

    fn coefficients_mut(&'a mut self) ->  &'a mut DVector<f64> {
        &mut self.plan.buf
    }

}

impl<'a, /*S*/ > Backward<'a, DVector<f64>, DVector<f64>> for DWT
where
    // S : ContiguousStorageMut<f64, Dynamic, U1>
{

    fn backward_from(&'a mut self, coefs : &'a DVector<f64>) -> Option<&'a DVector<f64>> {
        self.plan.buf.copy_from(&coefs);
        self.backward_from_self()
    }

    fn backward_from_self(&'a mut self) -> Option<&'a DVector<f64>> {
        if let Some(mut dom) = self.take_domain() {
            self.backward_mut(&mut dom);
            self.domain = Some(dom);
            self.domain()
        } else {
            None
        }
    }

    fn backward_mut(&self, dst : &mut DVector<f64>) {
        // let mut b_buf = self.domain.take().unwrap();
        self.plan.backward_to(dst);
        // self.domain = Some(b_buf);
        // self.back_call = true;
        // self.domain.as_ref().unwrap()
    }

    fn take_domain(&mut self) -> Option<DVector<f64>> {
        self.domain.take()
    }

    fn domain(&'a self) -> Option<&'a DVector<f64>> {
        self.domain.as_ref()
    }

    fn domain_mut(&'a mut self) -> Option<&'a mut DVector<f64>> {
        self.domain.as_mut()
    }
}

pub struct DWT2D {
    plan : DWTPlan2D,
    domain : Option<DMatrix<f64>>,
    // back_call : bool
}

impl DWT2D {

    pub fn new_empty(nrows : usize, ncols : usize) -> Self {
        Self::new(DMatrix::zeros(nrows, ncols))
    }

    pub fn new<S>(s : S) -> Self
        where S : Into<DMatrix<f64>>
    {
        let mut ms : DMatrix<f64> = s.into();
        let mut plan = DWTPlan2D::new(gsl::Basis::Daubechies(6, true), ms.nrows()).unwrap();
        let _ = plan.forward(&ms);
        let domain = Some(ms);
        //let back_call = true;
        //Self{ plan, domain, back_call }
        Self{ plan, domain }
    }

    pub fn iter_levels<'a>(&'a self) -> DWTIterator2D<'a> {
        self.plan.iter_levels()
    }

    pub fn iter_levels_mut<'a>(&'a mut self) -> DWTIteratorMut2D<'a> {
        self.plan.iter_levels_mut()
    }

}

impl<'a, S> Forward<'a, Matrix<f64, Dynamic, Dynamic, S>, DMatrix<f64>> for DWT2D
where S : ContiguousStorage<f64, Dynamic, Dynamic>
{

    fn forward_from(&'a mut self, s : &Matrix<f64, Dynamic, Dynamic, S>) -> &'a DMatrix<f64> {
        let _ = self.plan.forward(&s);
        &self.plan.buf
        // self.back_call = false;
        // &(*ans)
    }

    // fn partial_backward<S>(&'a mut self, n : usize) -> DMatrixSlice<'a, f64> {
    //    unimplemented!()
    // }

    fn coefficients(&'a self) -> &'a DMatrix<f64> {
        &self.plan.buf
    }

    fn coefficients_mut(&'a mut self) -> &'a mut DMatrix<f64> {
        &mut self.plan.buf
    }

}

impl<'a, /*S*/ > Backward<'a, DMatrix<f64>, DMatrix<f64>> for DWT2D
where
    //S : ContiguousStorageMut<f64, Dynamic, Dynamic>
{

    fn backward_from(&'a mut self, coefs : &'a DMatrix<f64>) -> Option<&'a DMatrix<f64>> {
        self.plan.buf.copy_from(&coefs);
        self.backward_from_self()
    }

    fn backward_from_self(&'a mut self) -> Option<&'a DMatrix<f64>> {
        if let Some(mut dom) = self.take_domain() {
            self.backward_mut(&mut dom);
            self.domain = Some(dom);
            self.domain()
        } else {
            None
        }
    }

    fn backward_mut(&self, dst : &mut DMatrix<f64>) {
        // let mut b_buf = self.domain.take().unwrap();
        self.plan.backward_to(dst);
        // self.domain = Some(b_buf);
        // self.back_call = true;
        // self.domain.as_ref().unwrap()
    }

    fn take_domain(&mut self) -> Option<DMatrix<f64>> {
        self.domain.take()
    }

    fn domain(&'a self) -> Option<&'a DMatrix<f64>> {
        self.domain.as_ref()
    }

    fn domain_mut(&'a mut self) -> Option<&'a mut DMatrix<f64>> {
        self.domain.as_mut()
    }
}*/


