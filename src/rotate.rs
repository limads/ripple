/// Rotates the signal by the given number of samples in the positive
/// or negative direction.
pub trait Rotate {

    fn rotate_to(&self, dst : &mut Self, by : i32);

}

