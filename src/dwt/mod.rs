#[cfg(feature="ipp")]
mod dwt1d;

#[cfg(feature="ipp")]
mod dwt2d;

#[cfg(feature="ipp")]
pub use dwt1d::*;

#[cfg(feature="ipp")]
pub use dwt2d::*;

// Filter coefficients were extracted from Tan & Jiang (2019) Digital Signal Processing
// Fundamentals and applications (Third Edition); Table 12.2 (p. 621).
pub mod bank {

    // Haar: 1 / sqrt(2) = 0.7...

    pub const HAAR_LOW : [f32; 2] = [1., 1.];

    pub const HAAR_HIGH : [f32; 2] = [-1., 1.];

    pub const HAAR_LOW_INV : [f32; 2] = [1., 1.];

    pub const HAAR_LOW_HIGH : [f32; 2] = [1., -1.];

    pub const DAUBECHIES : [f32; 4] = [
        0.48296291314453414337487159986,
        0.83651630373780790557529378092,
        0.22414386804201338102597276224,
        -0.12940952255126038117444941881
    ];

    /*pub const DAUB4_HIGH : [f32; 4] = [
        -0.12940952255126038117444941881,
        -0.22414386804201338102597276224,
        0.83651630373780790557529378092,
        -0.48296291314453414337487159986
    ];*/

    /* const fn implementation if const floating point arithmetic were to become stabilized
    // https://github.com/rust-lang/rust/issues/57241
    pub const fn define_coef<const U : usize>(h_k : &[f32; U], t : usize) -> f32 {
        let ix = ((U as i32 - 1 - t as i32) % U as i32) as usize;
        (-1. as f32).powf(t as f32) * h_k[ix]
    }

    pub const fn advance_level<const U : usize>(h_k : &[f32; 4]) -> [f32; 4] {
        [
            define_coef(h_k, 0),
            define_coef(h_k, 1),
            define_coef(h_k, 2),
            define_coef(h_k, 3)
        ]
    }*/

    // Start from a base filter h_k and generate the filter for level h_{k+1}
    pub fn advance_level<const U : usize>(h_k : &[f32; U]) -> [f32; U] {
        let mut h_kp1 = h_k.clone();
        for t in 0..U {
            let ix = ((U as i32 - 1 - t as i32) % U as i32) as usize;
            h_kp1[t] = (-1. as f32).powf(t as f32) * h_k[ix];
        }
        h_kp1
    }

}

#[test]
fn advance_level() {
    println!("Original: {:?}", &bank::DAUBECHIES);
    println!("Apply 1: {:?}", bank::advance_level(&bank::DAUBECHIES));
    println!("Apply 2: {:?}", bank::advance_level(&bank::advance_level(&bank::DAUBECHIES)));
    println!("Apply 3: {:?}", bank::advance_level(&bank::advance_level(&bank::advance_level(&bank::DAUBECHIES))));
    println!("Apply 4: {:?}", bank::advance_level(&bank::advance_level(&bank::advance_level(&bank::advance_level(&bank::DAUBECHIES)))));
}

#[derive(Clone, Debug)]
pub enum Basis {

    // Centered or not
    Haar(bool),

    // k = 4..20 even; centered or not
    Daubechies(usize, bool),

    // k = 103, 105, 202..208 even, 301..309 odd; centered or not
    BSpline(usize, bool)

}

impl Basis {

    fn len(&self) -> usize {
        match self {
            Basis::Daubechies(4, _) => 4,
            _ => unimplemented!()
        }
    }

    fn coefficients(&self) -> &'static [f32; 4] {
        match self {
            Basis::Daubechies(4, _) => &bank::DAUBECHIES,
            _ => panic!("Unimplemented basis {:?}", self)
        }
    }

    fn taps(&self, n_levels : usize) -> Vec<([f32; 4], [f32; 4])> {
        match self {
            Basis::Daubechies(4, _) => {
                let mut coefs = Vec::new();
                let low = self.coefficients().clone();
                let high = bank::advance_level(&low);
                coefs.push((low, high));
                for i in 1..(n_levels-1) {
                    let low = bank::advance_level(&coefs[i-1].1);
                    let high = bank::advance_level(&low);
                    coefs.push((low, high));
                }
                coefs
            },
            _ => unimplemented!()
        }
    }

}

