/// General-purpose iterators over dynamic matrices.
pub mod iter;

// Native discrete convolution.
//pub mod native;

/// Wrapper type to perform discrete convolution by binding against Intel MKL.
#[cfg(feature = "mkl")]
pub mod mkl;

#[derive(Clone, Copy, Debug)]
pub enum Padding {
    Extend,
    Wrap
}

pub fn convolve_with_array<const U : usize>(signal : &[f32], kernel : &[f32; U], out : &mut Vec<f32>, padding : Padding) {
    out.clear();
    signal.windows(U).for_each(|win| {
        let mut dot = 0.0;
        for i in 0..U {
            dot += win[i] * kernel[U-i-1];
        }
        out.push(dot);
    });
    match padding {
        Padding::Extend => {
            while out.len() != signal.len() {
                out.push(out[out.len() - 1]);
            }
        },
        _ => unimplemented!()
    }
}

