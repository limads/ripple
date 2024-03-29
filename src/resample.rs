use crate::signal::*;
use num_traits::Zero;
use nalgebra::Scalar;
use std::fmt::Debug;

pub enum Downsample {

    // Simply take every nth sample
    Aliased,

    // Apply gaussian smoothing prior to taking every nth sample.
    Smoothed

}

// Perhaps have a single structure resampling
// Upsample possibilities
pub enum Upsample {

    // Nearest-pixel interpolation (aliased)
    Nearest,

    // Linear interpolation
    Linear,

    // Polynomial interpolation
    Polynomial

}

pub trait Resample<O, M> {

    fn downsample_to(&self, out : &mut M, down : Downsample);

    fn upsample_to(&self, out : &mut M, up : Upsample);

    fn downsample(&self, down : Downsample, by : usize) -> O;

    fn upsample(&self, up : Upsample, by : usize) -> O;

}

impl<'a, N> Resample<Signal<N>, EpochMut<'a, N>> for Epoch<'a, N>
where
    N : Zero + Scalar + Copy + Debug
{

    fn downsample_to(&self, mut out : &mut EpochMut<'a, N>, down : Downsample) {

        let by = self.len() / out.len();
        assert!(out.len() % by == 0);

        match down {
            Downsample::Aliased => {
                // let step = src.slice.len() / self.buf.nrows();
                //if step == 1 {
                //    crate::signal::sampling::slices::convert_slice(
                //        &src.slice.as_slice(),
                //        self.buf.as_mut_slice()
                //    );
                // } else {
                /*let ncols = src.slice.ncols();
                assert!(src.slice.len() / step == self.buf.nrows(), "Dimension mismatch");
                crate::signal::sampling::slices::subsample_convert(
                    src.as_slice(),
                    out.as_mut_slice(),
                    ncols,
                    step,
                    false
                );*/
                // }

                // out.iter_mut().zip(self.iter().step_by(by)).for_each(|(dst, src)| *dst = *src );

                for ix in 0..out.len() {
                    out[ix] = self[ix*by]
                }

            },
            Downsample::Smoothed => {
                // Case smoothed, convolve with gaussian then re-call downsample_from with
                // mode aliased.
                unimplemented!()
            }
        }
    }

    fn upsample_to(&self, out : &mut EpochMut<'a, N>, up : Upsample) {
        unimplemented!()
    }

    fn downsample(&self, down : Downsample, by : usize) -> Signal<N> {
        // let mut out = Signal::new_constant(self.len() / by, N::zero());
        // out.full_epoch_mut().downsample_from(self, down, by);
        // out
        unimplemented!()
    }

    fn upsample(&self, up : Upsample, by : usize) -> Signal<N> {
        //let mut out = Signal::new_constant(self.len() / by, N::zero());
        //out.full_epoch_mut().upsample_from(self, up, by);
        //out
        unimplemented!()
    }

}


