/// Performs SIMD-accelerated sample-wise operations.
pub trait Packed {

    fn pow_to(&self, pow : i32, other : &mut Self);

    fn ln_to(&self, other : &mut Self);

    fn sqrt_to(&self, othere : &mut Self);

}

