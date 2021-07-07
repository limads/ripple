use std::ops::Range;

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

    pub const DAUB4_LOW : [f32; 4] = [
        0.48296291314453414337487159986,
        0.83651630373780790557529378092,
        0.22414386804201338102597276224,
        -0.12940952255126038117444941881
    ];

    pub const DAUB4_HIGH : [f32; 4] = [
        -0.12940952255126038117444941881,
        -0.22414386804201338102597276224,
        0.83651630373780790557529378092,
        -0.48296291314453414337487159986
    ];

    /*// const fn implementation if const floating point arithmetic were to become stabilized
    // https://github.com/rust-lang/rust/issues/57241
    pub const fn define_coef<const U : usize>(h_k : &[f32; U], t : usize) -> f32 {
        let ix = ((U as i32 - 1 - t as i32) % U as i32) as usize;
        (-1. as f32).powf(t as f32) * h_k[ix]
    }*/

    /*pub const fn advance_level<const U : usize>(h_k : &[f32; 4]) -> [f32; 4] {
        [
            define_coef(h_k, 0),
            define_coef(h_k, 1),
            define_coef(h_k, 2),
            define_coef(h_k, 3)
        ]
    }*/

    // Derive a difference filter from a coarse filter. This is used for testing purposes only,
    // since coefficients are passed as &'static [f32]. If const floating point arithmetic
    // is ever stabilized, we can use this to define the coefficients instead of using
    // hard-coded values.
    pub fn derive_diff<const U : usize>(h_k : &[f32; U]) -> [f32; U] {
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
    use crate::dwt::bank::*;
    println!("{:?} {:?}", derive_diff(&DAUBECHIES), DAUB4_HIGH);
    /*println!("Original: {:?}", &bank::DAUBECHIES);
    println!("Apply 1: {:?}", bank::derive_diff(&bank::DAUBECHIES));
    println!("Apply 2: {:?}", bank::derive_diff(&bank::derive_diff(&bank::DAUBECHIES)));
    println!("Apply 3: {:?}", bank::derive_diff(&bank::derive_diff(&bank::derive_diff(&bank::DAUBECHIES))));
    println!("Apply 4: {:?}", bank::derive_diff(&bank::derive_diff(&bank::derive_diff(&bank::derive_diff(&bank::DAUBECHIES)))));*/
}

#[derive(Clone, Copy, Debug)]
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

    fn coefficients(&self) -> (&'static [f32; 4], &'static [f32; 4]) {
        match self {
            Basis::Daubechies(4, _) => (&bank::DAUB4_LOW, &bank::DAUB4_HIGH),
            _ => panic!("Unimplemented basis {:?}", self)
        }
    }

    /*fn taps(&self, n_levels : usize) -> Vec<([f32; 4], [f32; 4])> {
        /*match self {
            Basis::Daubechies(4, _) => {
                let mut coefs = Vec::new();
                let low = self.coefficients().clone();
                let high = bank::derive_diff(&low);
                coefs.push((low, high));
                for i in 1..(n_levels-1) {
                    let low = bank::advance_level(&coefs[i-1].0);
                    let high = bank::advance_level(&coefs[i-1].1);
                    coefs.push((low, high));
                }
                coefs
            },
            _ => unimplemented!()
        }*/
        unimplemented!()
    }*/

}

// Checks if the vector with len given by the argument is a valid DWT argument.
// Returns true if it is an integer power of two. Returns false otherwise. Integer
// powers of two are required for DWT buffers because those are the only
// values that can be divided by two recursively until we arrive at a coarse
// coefficient buffer of size n=filter_len. For any other N even, we might only
// be able to perform one or a few steps of the decomposition, then fall into a value
// with odd number of elements, leaving the transform incomplete. Eventually, this might
// be all that the user needs, but we impose this restriction for now.
pub fn is_valid_dwt_len(len : usize) -> bool {
    len > 1 && (len as f64).log2().fract() == 0.0
}

// Checks if the side of the square image of given buffer len has a valid lenght for dwt decomposition.
pub fn is_valid_dwt_side(len : usize) -> bool {
    is_valid_dwt_len(side_len(len))
}

// Returns maximum number of levels for image of size n.
pub fn dwt_max_levels(n : usize) -> usize {
    (n as f32).log2() as usize - 1
}

// If cascade data is located in a contiguous buffer from coarse to detail, return the range that index
// the transformed buffer, assuming an original signal size n, and that we wish to index the level lvl.
fn dwt_level_index(n : usize, lvl : usize) -> Range<usize> {
    let ix = n / (2usize).pow(lvl as u32 + 1);
    ix..2*ix
}

fn verify_dwt_dimensions(src_len : usize, coarse_len : usize, detail_len : usize) {
    assert!(src_len / 2 == coarse_len);
    assert!(src_len / 2 == detail_len);
    for len in [src_len, coarse_len, detail_len].iter() {
        assert!(is_valid_dwt_len(*len));
    }
}

fn verify_dwt1d_dimensions(src : &[f32], coarse : &[f32], detail : &[f32]) {
    verify_dwt_dimensions(src.len(), coarse.len(), detail.len());
}

fn verify_dwt2d_dimensions(src : &[f32], coarse : &[f32], detail_x : &[f32], detail_y : &[f32], detail_xy : &[f32]) {
    verify_dwt_dimensions(side_len(src.len()), side_len(coarse.len()), side_len(detail_x.len()));
    verify_dwt_dimensions(side_len(src.len()), side_len(coarse.len()), side_len(detail_y.len()));
    verify_dwt_dimensions(side_len(src.len()), side_len(coarse.len()), side_len(detail_xy.len()));
}

/// Tests len is a power of two and returns the side at the same time.
fn side_len(len : usize) -> usize {
    let sq_len = (len as f32).sqrt();
    assert!(sq_len.fract() == 0.0);
    sq_len as usize
}


fn index_and_prev_mut<T>(levels : &mut [T], last : usize) -> (&mut T, &mut T) {
    assert!(last >= 1);
    let lvls = levels[last-1..last+1].split_at_mut(1);
    let (prev_lvl, curr_lvl) = (&mut lvls.0[0], &mut lvls.1[1]);
    (prev_lvl, curr_lvl)
}
