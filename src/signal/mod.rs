use nalgebra::*;
use nalgebra::storage::*;
use std::ops::{Index, Mul, Add, AddAssign, MulAssign, Div, SubAssign, Sub};
use simba::scalar::SubsetOf;
use simba::scalar::SupersetOf;
use std::fmt::Debug;
use std::fmt::{self, Display};
use serde::Deserialize;
use std::convert::TryFrom;
use serde::Deserializer;
use std::iter::{FromIterator, Extend, IntoIterator};
use std::cmp::PartialEq;
use std::ops::Range;
use num_traits::{Float, Zero};
use std::borrow::{Borrow, ToOwned};
use std::ops::IndexMut;

pub mod sampling;

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
    N : Scalar + Copy
{

    pub fn new_constant(n : usize, value : N) -> Self {
        Self{ buf : DVector::from_element(n, value) }
    }

    pub fn full_epoch(&'a self) -> Epoch<'a, N> {
        Epoch{ slice : self.buf.rows(0, self.buf.nrows()), offset : 0 }
    }

    pub fn full_epoch_mut(&'a mut self) -> EpochMut<'a, N> {
        EpochMut{ slice : self.buf.rows_mut(0, self.buf.nrows()), offset : 0 }
    }

    pub fn as_slice(&self) -> &[N] {
        self.as_ref()
    }

    pub fn len(&self) -> usize {
        self.buf.nrows()
    }

    pub fn epoch(&'a self, start : usize, len : usize) -> Epoch<'a, N> {
        assert!(start + len <= self.buf.nrows());
        Epoch{ slice : self.buf.rows(start, len), offset : start }
    }

    pub fn epoch_mut(&'a mut self, start : usize, len : usize) -> EpochMut<'a, N> {
        assert!(start + len <= self.buf.nrows());
        EpochMut{ slice : self.buf.rows_mut(start, len), offset : start }
    }

}

impl<'a, N> Signal<N>
where
    N : Scalar + Copy + MulAssign + AddAssign + Add<Output=N> + Mul<Output=N> + SubAssign + Field + SimdPartialOrd + From<f32>,
    f64 : SubsetOf<N>
{

    pub fn delayed_clone(&self, n : usize) -> Self {
        let cloned = self.clone();
        cloned.delay(n)
    }

    fn advanced_clone(&self, n :  usize) -> Self {
        let cloned = self.clone();
        cloned.advance(n)
    }

    pub fn delay(mut self, n : usize) -> Self {
        self.circ_shift(n, Direction::Delay)
    }

    pub fn advance(mut self, n : usize) -> Self {
        self.circ_shift(n, Direction::Advance)
    }

    /*
    pub fn threshold(&self)
        IppStatus ippsThreshold_16s(const Ipp16s* pSrc, Ipp16s* pDst, int len, Ipp16s level,
        IppCmpOp relOp );
    */

    /*
        pub fn flip()
    */

    // autocorrelation(.)
    // The ith autocorrelation sample is the product of the centered signal
    // with the same centered signal shifted by i.

    // autoregressive(order : usize)
    // Calculates the autoregressive estimates using the Yule-Walker estimate.

    // Calculate signal distance for each possible signal shift. Returns the
    // signal shift with the smallest distance.
    // fn align().

    // Gaussian process or spline-based signal interpolation.
    // interpolate()

    // Calculates a smooth envelope over rapidly-varying signal regions (e.g. EMG data).
    // (1) Square each centered sample in the signal
    // (2) Smooth the squared samples with window of length N
    // (3) Take the square root of each sample.
    // The peaks can then be found by template-based or geometrical-based algorithms.
    // rms_envelope(.)

    // fn template_search
    // (1) Correlate the signal with a few desired templates (invert the template then convolve)
    // (2) Take local maxima of the cross-correlation.

    /// Advance : Push Signal to origin; Delay : pulls Signal away from origin. TODO move to Shift
    /// trait, that is also implemented for images.
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

    /*
        Convert
        IppStatus ippsConvert_8s16s(const Ipp8s* pSrc,Ipp16s* pDst, int len );
    */

    // Divide by signal "energy" (variance).
    /*pub fn normalize(&mut self) {
        IppStatus ippsNormalize_32f(const Ipp32f* pSrc, Ipp32f* pDst, int len, Ipp32f vSub,
        Ipp32f vDiv );
    }*/

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

    // Iterate over epochs of same size.
    // pub fn epochs(&self, size : usize) -> impl Iterator<Item=Epoch<'_, N>> {
    //    unimplemented!()
    // }

    // Move this to method of Pyramid.
    // pub fn threshold(&self, thr : &Threshold) -> SparseSignal<'_, N> {
    //    unimplemented!()
    // }
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
    N : Scalar + Copy,
{
    fn as_ref(&self) -> &[N] {
        self.slice.as_slice()
    }
}

impl<N> AsMut<[N]> for EpochMut<'_, N>
where
    N : Scalar + Copy
{
    fn as_mut(&mut self) -> &mut [N] {
        self.slice.as_mut_slice()
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

impl<N> Extend<N> for Signal<N>
where
    N : Scalar + Copy + MulAssign + AddAssign + Add<Output=N> + Mul<Output=N> + SubAssign + Field + SimdPartialOrd + From<f32>,
    f64 : SubsetOf<N>
{

    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = N>
    {
        let mut v = self.buf.data.as_vec().clone();
        v.extend(iter);
        self.buf = DVector::from_vec(v);
    }

}

impl<N> IntoIterator for Signal<N>
where
    N : Scalar
{
    type Item = N;

    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(mut self) -> Self::IntoIter {
        let data : Vec<_> = self.buf.data.into();
        data.into_iter()
    }

}

/// Borrowed subset of a Signal.
#[derive(Debug, Clone)]
pub struct Epoch<'a, N>
where
    N : Scalar
{
    offset : usize,
    slice : DVectorSlice<'a, N>
}

impl<'a, N> Epoch<'a, N>
where
    N : Scalar + Copy
{

    pub fn clone_owned(&self) -> Signal<N> {
        Signal::from(self.slice.as_slice()[self.offset..].to_vec())
    }

    pub fn len(&'a self) -> usize {
        self.slice.nrows()
    }

    pub fn iter(&self) -> impl Iterator<Item=&N> {
        self.slice.iter()
    }

    /// Returns labeled samples, with indices refering to the indices of the subsampled signal.
    pub fn labeled_samples(&self, spacing : usize) -> impl Iterator<Item=(usize, &N)> {
        self.iter().step_by(spacing).enumerate()
    }

    pub fn sub_epoch(&'a self, pos : usize, len : usize) -> Epoch<'a, N> {
        Self { slice : DVectorSlice::from(&self.slice.as_slice()[pos..pos+len]), offset : self.offset + pos }
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

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
    N : Scalar + Copy {

    pub fn sub_epoch_mut(&mut self, pos : usize, len : usize) -> EpochMut<'a, N> {
        // Self { slice : DVectorSliceMut::from(&mut self.slice.as_mut_slice()[pos..pos+len]), offset : self.offset + pos }
        unimplemented!()
    }

    pub fn len(&self) -> usize {
        self.slice.len()
    }

    pub fn iter_mut(&'a mut self) -> impl Iterator<Item=&'a mut N> {
        self.slice.iter_mut()
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

}

impl<'a, N> EpochMut<'a, N>
where
    N : Scalar + Copy + MulAssign + AddAssign + Add<Output=N> + Mul<Output=N> + SubAssign + Field + SimdPartialOrd,
    f64 : SubsetOf<N>
{

    pub fn clone_owned(&self) -> Signal<N> {
        // let v : Vecself.slice.clone()
        unimplemented!()
    }

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

}

impl<'a, N> Epoch<'a, N>
where
    N : Scalar + Copy + MulAssign + AddAssign + Add<Output=N> + Mul<Output=N> + SubAssign + Field + SimdPartialOrd + From<f32> + Float + Zero + Div<Output=N>,
    f64 : SubsetOf<N>
{

    pub fn std_dev(&self) -> N {
        self.variance().sqrt()
    }

    // Applicable to non-floating-point signals, unlike std_dev. Also in the
    // same units of measurement as the original signal.
    pub fn abs_dev(&self) -> N {
        let mean = self.slice.mean();
        self.slice.iter().map(|s| (*s - mean).abs() ).fold(N::zero(), |acc, s| acc + s )
    }

    /*// Smith (1997) p. 17
    pub fn coefficient_variation(&self) -> N {
        (From::from(1. as f32) / self.snr()) * From::from(100.0 as f32).
    }*/

    /// Returns mean / stddev
    pub fn snr(&self) -> N {
        self.mean() / self.std_dev()
    }

    /// Returns maximum peak-to-peak amplitude at the signal.
    pub fn max_amplitude(&self) -> N {
        let max = self.max();
        let min = self.min();
        max.abs() + min.abs()
    }

}

#[derive(Clone, Copy, Debug)]
pub struct Overlap {
    pub middle : usize,
    pub last : usize
}

impl<'a, N> Epoch<'a, N>
where
    N : Scalar + Copy + MulAssign + AddAssign + Add<Output=N> + Mul<Output=N> + SubAssign + Field + SimdPartialOrd + From<f32> + std::cmp::PartialOrd,
    f64 : SubsetOf<N>
{

    // TODO envelope extraction (hilbert)
    // TODO slewrate

    // Returns the non-final nonoverlapping epochs, ignoring the signal end.
    pub fn nonoverlapping_epochs(&'a self, epoch_len : usize) -> Vec<Epoch<'a, N>>
    where
        N : Scalar + Copy + MulAssign + AddAssign + Add<Output=N> + Mul<Output=N> + SubAssign + Field + SimdPartialOrd + From<f32>,
        f64 : SubsetOf<N>
    {
        assert!(self.len() % epoch_len == 0);
        let mut epochs = Vec::new();
        for ix in 0..(self.len() / epoch_len) {
            epochs.push(self.sub_epoch(ix*epoch_len, epoch_len));
        }
        epochs
    }


    /// Search the signal for the longest where the given condition is met.
    /// At each iteration, reduce the slice length by step_len.
    pub fn longest_matching_region<F>(
        &'a self,
        min_sz : usize,
        step_len : usize,
        condition : F
    ) -> Option<Epoch<'a, N>>
    where
        F : Fn(Epoch<'a, N>)->bool
    {
        assert!(min_sz > 2);
        assert!(step_len > 1);
        let mut seq_len = self.len();
        loop {
            for sub_epoch in self.overlapping_epochs(seq_len) {
                assert!(sub_epoch.offset() >= self.offset && sub_epoch.len() <= self.len());
                assert!(sub_epoch.slice.len() == sub_epoch.len());
                if condition(sub_epoch.clone()) {
                    let offset_diff = (sub_epoch.offset() - self.offset());
                    let len_diff = self.len() - sub_epoch.len();
                    assert!(offset_diff + sub_epoch.len() + (self.end_offset() - sub_epoch.end_offset()) == self.len());
                    return Some(sub_epoch);
                }
            }
            if seq_len as i32 - step_len as i32 > min_sz as i32 {
                seq_len -= step_len;
            } else {
                return None;
            }
        }
        None
    }

    /// Returns one-past index of underlying buffer corresponding to last sample.
    pub fn end_offset(&self) -> usize {
        self.offset() + self.len()
    }

    pub fn overlapping_epochs(&'a self, epoch_len : usize) -> impl Iterator<Item=Epoch<'a, N>> {
        self.slice.as_slice().windows(epoch_len)
            .enumerate()
            .map(move |(ix, win)| {
                // Create new epoch by converting from slice (offset=0 here)
                let mut e = Epoch::from(win);

                // Update the offset
                e.offset = self.offset + ix;

                // println!("{:?}", e);
                e
            })
    }

    /// Returns a sequence of minimally-overlapping epochs of size len; and the size of the k-1 overlaps
    /// of all epochs after the first. All samples of the signal will be returned, but samples will be
    // repeated at the beginning and end of each epoch.
    pub fn minimally_overlapping_epochs(&'a self, epoch_len : usize) -> (Vec<Epoch<'a, N>>, Overlap)
    where
        N : Scalar + Copy + MulAssign + AddAssign + Add<Output=N> + Mul<Output=N> + SubAssign + Field + SimdPartialOrd + From<f32>,
        f64 : SubsetOf<N>
    {
        assert!(self.len() >= epoch_len);
        let full_len = self.len();
        let mut gap = self.len() % epoch_len;
        let mut epochs = Vec::new();

        if self.len() == epoch_len {
            epochs.push(self.clone());
            return(epochs, Overlap { middle : 0, last : 0 });
        }

        if gap == 0 {
            epochs = self.nonoverlapping_epochs(epoch_len);
            return(epochs, Overlap { middle : 0, last : 0 });
        }

        // Number of nonfinal epochs, including the first
        let mut n_nonfinal_epochs = self.len() / epoch_len;

        // println!("n_nonfinal: {}", n_nonfinal_epochs);
        // println!("gap: {}", gap);

        let middle_overlap = (epoch_len - gap) / (n_nonfinal_epochs);

        // println!("middle overlap: {}", middle_overlap);
        let mut offset = 0;
        for i in 0..n_nonfinal_epochs {
            let curr_epoch = self.sub_epoch(offset, epoch_len);
            // println!("Curr epoch: {:?}", curr_epoch);
            epochs.push(curr_epoch);
            offset += epoch_len - middle_overlap;
            // println!("Next offset = {}", offset);
        }
        let last_nonfinal_offset = epochs[epochs.len()-1].offset();

        let last_offset = full_len - epoch_len;
        let last_epoch = self.sub_epoch(last_offset, epoch_len);
        let last_overlap = (last_nonfinal_offset + epoch_len).saturating_sub(last_epoch.offset());
        // println!("Last epoch = {:?}", last_epoch);
        epochs.push(last_epoch);
        let overlap = Overlap { middle : middle_overlap, last : last_overlap };

        if epochs.len() == 2 {
            assert!(overlap.middle == overlap.last);
        }

        // println!("Overlap = {:?}", overlap);
        // assert!( (n_epochs-1)*epoch_len - gap + epoch_len == s.len() );
        (epochs, overlap )
    }

    /// Returns the difference signal, with size (n-1) / k, where k is the size of the window.
    /// The pth order difference (i.e. for a position signal,
    /// returns velocity, then acceleration) can be built by recursively calling this function.
    /// Calling this function equals convolving with the [-1, 0, ... 0, 1] difference filter.
    pub fn local_difference(&self) -> Signal<N> {
        let mut diffs = Vec::new();
        for ix in 1..self.slice.nrows() {
            diffs.push(self.slice[ix-1] - self.slice[ix]);
        }
        Signal::from(diffs)
    }

    /// Calling this function equals convolving with the (1/k)*[1, 1, ... 1, 1] smoothing filter.
    pub fn local_average(&self) -> Signal<N> {
        let mut diffs = Vec::new();
        for ix in 1..self.slice.nrows() {
            diffs.push((self.slice[ix-1] + self.slice[ix]) * N::from(0.5));
        }
        Signal::from(diffs)
    }

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

}


#[test]
fn overlap() {
    let s = Signal::from_iter((0..33).map(|s| s as f64) );
    s.full_epoch().minimally_overlapping_epochs(10);
}

impl<N> Index<usize> for Signal<N>
where
    N : Scalar
{

    type Output = N;

    fn index(&self, ix: usize) -> &N {
        &self.buf[ix]
    }
}

impl<'a, N> Index<usize> for Epoch<'a, N>
where
    N : Scalar
{

    type Output = N;

    fn index(&self, ix: usize) -> &N {
        &self.slice[ix]
    }
}

impl<'a, N> Index<usize> for EpochMut<'a, N>
where
    N : Scalar
{

    type Output = N;

    fn index(&self, ix: usize) -> &N {
        &self.slice[ix]
    }

}

impl<'a, N> IndexMut<usize> for EpochMut<'a, N>
where
    N : Scalar
{

    // type Output = N;

    fn index_mut(&mut self, ix: usize) -> &mut N {
        &mut self.slice[ix]
    }

}

/*impl<'a, N> Index<Range<usize>> for Epoch<'a, N>
where
    N : Scalar + Copy + MulAssign + AddAssign + Add<Output=N> + Mul<Output=N> + SubAssign + Field + SimdPartialOrd,
    f64 : SubsetOf<N>
{

    type Output = Epoch<'a, N>;

    fn index(&self, ix: Range<usize>) -> Epoch<'a, N> {
        self.sub_epoch(ix.start, ix.end - ix.start)
    }
}*/

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
        unsafe{ self.buf.data.into() }
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

    /// Generate an impulse signal.
    pub fn pulse<T>(n : usize) -> Signal<T>
    where
        T : From<f32> + Scalar + Div<Output=T> + Copy + Debug
    {
        let mut s = flat::<T>(n);
        let s_slice : &mut [T] = s.as_mut();
        s_slice[0] = T::from(1.0);
        s
    }

    /// Generate white noise.
    pub fn white<T>(n : usize) -> Signal<T>
    where
        T : From<f64> + Scalar + Div<Output=T> + Copy + Debug
    {
        // use rand::prelude::*;
        // the trait `rand::distributions::Distribution<_>` is not implemented for `rand_distr::StandardNormal`
        use rand::Rng;
        use rand::distributions::Distribution;
        let mut rng = rand::thread_rng();
        /*(0..n).map(|_| {
            let z : f64 = rng.sample(rand_distr::StandardNormal);
            T::from(z)
        }).collect()*/
        unimplemented!()
    }

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

    pub fn step2d<T>(side : usize) -> Signal<T>
    where
        T : Scalar + Copy + MulAssign + AddAssign + Add<Output=T> + Mul<Output=T> + SubAssign + Field + SimdPartialOrd + From<f32>,
        f64 : SubsetOf<T>
    {
        let half_len = side / 2;
        let mut img = flat::<T>(side);
        for i in 0..(side-1) {
            if i <= (half_len-1) {
                img.extend(flat(side));
            } else {
                img.extend(step(side));
            }
        }
        img
    }

    // IppStatus ippsWinBartlett_16s(const Ipp16s* pSrc, Ipp16s* pDst, int len );
    // IppStatus ippsWinBlackman_16s(const Ipp16s* pSrc, Ipp16s* pDst, int len, Ipp32f alpha );
    // IppStatus ippsWinHamming_16s(const Ipp16s* pSrc, Ipp16s* pDst, int len );
    // IppStatus ippsWinHann_16s(const Ipp16s* pSrc, Ipp16s* pDst, int len );
    // IppStatus ippsWinKaiser_16s(const Ipp16s* pSrc, Ipp16s* pDst, int len, Ipp32f alpha );
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

/*impl<'a, N> Borrow<Epoch<'a, N>> for Signal<N> {

    fn borrow(&self) -> &Epoch<'a, N> {

    }

}

pub trait ToOwned {

    type Owned = ;

    fn to_owned(&self) -> Self::Owned;

    fn clone_into(&self, target: &mut Self::Owned) { ... }

}*/


