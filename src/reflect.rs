/// Reflects a signal along its midpoint.
pub trait Reflect {

    fn reflect_to(&self, other : &mut Self);

}

