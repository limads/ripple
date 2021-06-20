use nalgebra::*;
use nalgebra::storage::*;
use std::ops::{Index, Mul, Add, AddAssign, MulAssign, Div, SubAssign};
pub mod sampling;
use simba::scalar::SubsetOf;
use simba::scalar::SupersetOf;
use std::fmt::Debug;
use std::fmt::{self, Display};
use serde::Deserialize;
use std::convert::TryFrom;
use serde::Deserializer;
use std::iter::FromIterator;

pub mod conv;

/// Owned time Signal data structure.
#[derive(Debug, Clone)]
pub struct Signal<N>
where
    N : Scalar
{
    pub(crate) buf : DVector<N>
}

impl<'de, N> Deserialize<'de> for Signal<N>
where
    N : Scalar + Deserialize<'de>
{

    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>
    {
        let v : Vec<N> = Deserialize::deserialize(deserializer)?;
        Ok(Signal { buf : DVector::from_vec(v) })
    }

}

pub enum Direction {
    Advance,
    Delay
}

impl<'a, N> Signal<N>
where
    N : Scalar + Copy + MulAssign + AddAssign + Add<Output=N> + Mul<Output=N> + SubAssign + Field + SimdPartialOrd + From<f32>,
    f64 : SubsetOf<N>
{

    pub fn as_slice(&self) -> &[N] {
        self.as_ref()
    }

    pub fn len(&self) -> usize {
        self.buf.nrows()
    }

    pub fn delay(mut self, n : usize) -> Self {
        self.circ_shift(n, Direction::Delay)
    }


    pub fn advance(mut self, n : usize) -> Self {
        self.circ_shift(n, Direction::Advance)
    }

    /// Advance : Push Signal to origin; Delay : pulls Signal away from origin.
    fn circ_shift(mut self, n : usize, dir : Direction) -> Self {
        let mut tmp = Vec::new();
        let sz = self.buf.nrows();
        let sz_compl = sz - n;
        let tmp_sz = match dir {
            Direction::Advance => n,
            Direction::Delay => sz - n
        };
        tmp.extend((0..tmp_sz).map(|_| N::from(0. as f32) ));

        match dir {
            Direction::Advance => {

                // Saves the beginning part that will be written over
                tmp.copy_from_slice(&self.buf.as_slice()[0..n]);

                // Resolve the actual shift
                self.buf.as_mut_slice().copy_within(n..sz, 0);

                // Copied the saved beginning to the end
                self.buf.as_mut_slice()[sz-n..].copy_from_slice(&tmp[..]);
            },
            Direction::Delay => {
                unimplemented!()
            }
        }
        self
    }

    pub fn new_constant(n : usize, value : N) -> Self {
        Self{ buf : DVector::from_element(n, value) }
    }

    pub fn full_epoch(&'a self) -> Epoch<'a, N> {
        Epoch{ slice : self.buf.rows(0, self.buf.nrows()), offset : 0 }
    }

    pub fn full_epoch_mut(&'a mut self) -> EpochMut<'a, N> {
        EpochMut{ slice : self.buf.rows_mut(0, self.buf.nrows()), offset : 0 }
    }

    pub fn epoch(&'a self, start : usize, len : usize) -> Epoch<'a, N> {
        Epoch{ slice : self.buf.rows(start, len), offset : start }
    }

    pub fn epoch_mut(&'a mut self, start : usize, len : usize) -> EpochMut<'a, N> {
        EpochMut{ slice : self.buf.rows_mut(start, len), offset : start }
    }

    pub fn iter(&self) -> impl Iterator<Item=&N> {
        self.buf.iter()
    }

    pub fn iter_mut(&'a mut self) -> impl Iterator<Item=&'a mut N> {
        self.buf.iter_mut()
    }

    pub fn mean(&mut self) -> N {
        self.full_epoch_mut().mean()
    }

    pub fn offset_by(&mut self, scalar : N) {
        self.full_epoch_mut().offset_by(scalar)
    }

    pub fn downsample_aliased(&mut self, src : &Epoch<'_, N>) {
        let step = src.slice.len() / self.buf.nrows();
        if step == 1 {
            sampling::slices::convert_slice(
                &src.slice.as_slice(),
                self.buf.as_mut_slice()
            );
        } else {
            let ncols = src.slice.ncols();
            assert!(src.slice.len() / step == self.buf.nrows(), "Dimension mismatch");
            sampling::slices::subsample_convert(
                src.slice.as_slice(),
                self.buf.as_mut_slice(),
                ncols,
                step,
                false
            );
        }
    }

    // Iterate over epochs of same size.
    // pub fn epochs(&self, size : usize) -> impl Iterator<Item=Epoch<'_, N>> {
    //    unimplemented!()
    // }

    /*pub fn downsample_from(&mut self, other : &Self) {
        unimplemented!()
    }

    pub fn downsample_into(&self, other : &mut Self) {
        unimplemented!()
    }

    pub fn upsample_from(&mut self, other : &Self) {
        unimplemented!()
    }

    pub fn upsample_into(&self, other : &mut Self) {
        unimplemented!()
    }

    // Move this to method of Pyramid.
    pub fn threshold(&self, thr : &Threshold) -> SparseSignal<'_, N> {
        unimplemented!()
    }*/
}

impl<'a, N> From<&'a [N]> for Epoch<'a, N>
where
    N : Scalar + Copy + MulAssign + AddAssign + Add<Output=N> + Mul<Output=N> + SubAssign + Field + SimdPartialOrd,
    f64 : SubsetOf<N>
{

    fn from(slice : &'a [N]) -> Self {
        Self { slice : DVectorSlice::from(slice), offset : 0 }
    }

}

impl<'a, N> From<DVectorSlice<'a, N>> for Epoch<'a, N>
where
    N : Scalar + Copy + MulAssign + AddAssign + Add<Output=N> + Mul<Output=N> + SubAssign + Field + SimdPartialOrd,
    f64 : SubsetOf<N>
{

    fn from(slice : DVectorSlice<'a, N>) -> Self {
        Self { slice, offset : 0 }
    }

}

impl<N> AsRef<[N]> for Epoch<'_, N>
where
    N : Scalar + Copy + MulAssign + AddAssign + Add<Output=N> + Mul<Output=N> + SubAssign + Field + SimdPartialOrd,
    f64 : SubsetOf<N>
{
    fn as_ref(&self) -> &[N] {
        self.slice.as_slice()
    }
}

impl<N> FromIterator<N> for Signal<N>
where
    N : Scalar
{

    fn from_iter<I : IntoIterator<Item=N>>(iter: I) -> Self {
        let buf = DVector::from(Vec::from_iter(iter));
        Self { buf }
    }

}

/*impl<N> Index<(usize, usize)> for Image<N>
where
    N : Scalar
{

    type Output = N;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.buf[index]
    }
}

impl<N> IndexMut<(usize, usize)> for Image<N>
where
    N : Scalar
{

    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.buf[index]
    }

}*/

/// Borrowed subset of a Signal.
#[derive(Debug, Clone)]
pub struct Epoch<'a, N>
where
    N : Scalar
{
    offset : usize,
    slice : DVectorSlice<'a, N>
}



#[derive(Debug)]
pub struct EpochMut<'a, N>
where
    N : Scalar
{
    offset : usize,
    slice : DVectorSliceMut<'a, N>
}

impl<'a, N> EpochMut<'a, N>
where
    N : Scalar + Copy + MulAssign + AddAssign + Add<Output=N> + Mul<Output=N> + SubAssign + Field + SimdPartialOrd,
    f64 : SubsetOf<N>
{

    pub fn sum(&self) -> N {
        self.slice.sum()
    }

    pub fn mean(&self) -> N {
        self.slice.mean()
    }

    pub fn max(&self) -> N {
        self.slice.max()
    }

    pub fn min(&self) -> N {
        self.slice.min()
    }

    pub fn len(&self) -> usize {
        self.slice.len()
    }

    pub fn component_add(&mut self, other : &Epoch<N>) {
        self.slice.add_assign(&other.slice);
    }

    pub fn component_sub(&mut self, other : &Epoch<N>) {
        self.slice.sub_assign(&other.slice);
        //unimplemented!()
    }

    pub fn component_scale(&mut self, other : &Epoch<N>) {
        self.slice.component_mul_assign(&other.slice);
    }

    pub fn offset_by(&mut self, scalar : N) {
        self.slice.add_scalar_mut(scalar);
    }

    pub fn scale_by(&mut self, scalar : N) {
        // self.slice.scale_mut(scalar); // Only available for owned versions
        self.slice.iter_mut().for_each(|n| *n *= scalar );
    }

    pub fn iter_mut(&'a mut self) -> impl Iterator<Item=&'a mut N> {
        self.slice.iter_mut()
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

}

impl<'a, N> Epoch<'a, N>
where
    N : Scalar + Copy + MulAssign + AddAssign + Add<Output=N> + Mul<Output=N> + SubAssign + Field + SimdPartialOrd,
    f64 : SubsetOf<N>
{

    pub fn max(&self) -> N {
        self.slice.max()
    }

    pub fn min(&self) -> N {
        self.slice.min()
    }

    pub fn sum(&self) -> N {
        self.slice.sum()
    }

    pub fn mean(&self) -> N {
        self.slice.mean()
    }

    pub fn variance(&self) -> N {
        self.slice.variance()
    }

    pub fn len(&'a self) -> usize {
        self.slice.len()
    }

    pub fn iter(&self) -> impl Iterator<Item=&N> {
        self.slice.iter()
    }

    pub fn sub_epoch(&'a self, pos : usize, len : usize) -> Epoch<'a, N> {
        Self { slice : DVectorSlice::from(&self.slice.as_slice()[pos..pos+len]), offset : self.offset + pos }
    }

}

/*impl<'a, M, N> Downsample<Signal<N>> for Epoch<'a, M>
where
    M : Scalar + Copy,
    N : Scalar + Copy + From<M>
{

    fn downsample_aliased(&self, dst : &mut Signal<N>) {
        let step = self.slice.len() / dst.buf.nrows();
        if step == 1 {
            sampling::slices::convert_slice(
                &self.slice.as_slice()[self.offset..],
                dst.buf.as_mut_slice()
            );
        } else {
            let ncols = dst.buf.ncols();
            assert!(self.slice.len() / step == dst.buf.nrows(), "Dimension mismatch");
            sampling::slices::subsample_convert(
                self.slice.as_slice(),
                dst.buf.as_mut_slice(),
                ncols,
                step,
                false
            );
        }
    }
}*/

impl<N> Index<usize> for Signal<N>
where
    N : Scalar
{

    type Output = N;

    fn index(&self, ix: usize) -> &N {
        &self.buf[ix]
    }
}

impl<N> From<DVector<N>> for Signal<N>
where
    N : Scalar
{
    fn from(s : DVector<N>) -> Self {
        Self{ buf : s }
    }
}

impl<N> From<Vec<N>> for Signal<N>
where
    N : Scalar
{
    fn from(s : Vec<N>) -> Self {
        Self{ buf : DVector::from_vec(s) }
    }
}

impl<N> Into<Vec<N>> for Signal<N>
where
    N : Scalar
{
    fn into(self) -> Vec<N> {
        let n = self.buf.nrows();
        unsafe{ self.buf.data.resize(n) }
    }
}

impl<N> AsRef<[N]> for Signal<N>
where
    N : Scalar
{
    fn as_ref(&self) -> &[N] {
        self.buf.data.as_slice()
    }
}

impl<N> AsMut<[N]> for Signal<N>
where
    N : Scalar
{
    fn as_mut(&mut self) -> &mut [N] {
        self.buf.data.as_mut_slice()
    }
}

impl<N> AsRef<Vec<N>> for Signal<N>
where
    N : Scalar
{
    fn as_ref(&self) -> &Vec<N> {
        self.buf.data.as_vec()
    }
}

impl<N> AsMut<Vec<N>> for Signal<N>
where
    N : Scalar
{
    fn as_mut(&mut self) -> &mut Vec<N> {
        unsafe{ self.buf.data.as_vec_mut() }
    }
}

impl<N> AsRef<DVector<N>> for Signal<N>
where
    N : Scalar
{
    fn as_ref(&self) -> &DVector<N> {
        &self.buf
    }
}

impl<N> AsMut<DVector<N>> for Signal<N>
where
    N : Scalar
{
    fn as_mut(&mut self) -> &mut DVector<N> {
        &mut self.buf
    }
}

pub struct Threshold {

    /// Minimum distance of neighboring values (in number of samples
    /// or symmetrical pixel area). If None, all values satisfying value
    /// will be accepted.
    pub min_dist : Option<usize>,

    /// Threshold value.
    pub value : f64,

    /// All neighboring pixels over the area should be smaller by the
    /// specified ratio. If none, no slope restrictions are imposed.
    pub slope : Option<f64>
}

/*/// Result of thresholding a Signal. This structure carries a (row, col) index and
/// a scalar value for this index.
pub struct SparseSignal<'a, N>
where
    N : Scalar
{
    src : &'a Signal<N>
}

/// Subset of a SparseSignal
pub struct SparseEpoch<'a, N>
where
    N : Scalar
{

    /// Source sparse Signal
    source : &'a SparseSignal<'a, N>,

    /// Which indices we will use from the source
    ixs : Vec<usize>
}*/

/// Auto-generated Signals
pub mod gen {

    use super::*;

    pub fn flat<T>(n : usize) -> Signal<T>
    where
        T : From<f32> + Scalar + Div<Output=T> + Copy + Debug
    {
        let max = T::from(n as f32);
        Signal { buf : DVector::from_iterator(n, (0..n).map(|_| T::from(0.) )) }
    }

    pub fn ramp<T>(n : usize) -> Signal<T>
    where
        T : From<f32> + Scalar + Div<Output=T> + Copy + Debug
    {
        let max = T::from(n as f32);
        Signal { buf : DVector::from_iterator(n, (0..n).map(|s| T::from(s as f32) / max )) }
    }

    pub fn step<T>(n : usize) -> Signal<T>
    where
        T : From<f32> + Scalar  + Div<Output=T> + Copy + Debug
    {
        let half_max = n / 2;
        let step_vec = DVector::from_iterator(
            n,
            (0..n).map(|ix| if ix < half_max { T::from(0 as f32) } else { T::from(1.) })
        );
        Signal {
            buf : step_vec
        }
    }

}

impl<T> Add for Signal<T>
where
    T : Scalar + AddAssign + Copy
{
    type Output = Self;

    fn add(mut self, other: Self) -> Self {
        self.buf.iter_mut().zip(other.buf.iter()).for_each(|(s, o)| *s += *o );
        self
    }
}

impl<T> fmt::Display for Signal<T>
where
    T : Debug + Display + Copy + Debug + PartialEq + 'static
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.buf)
    }
}


