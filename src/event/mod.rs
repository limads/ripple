use std::f32::consts::PI;
use std::cmp::{PartialOrd, Ordering};
use nalgebra::Scalar;
use num_traits::real::Real;
use num_traits::Zero;

pub struct Event<N> {
    pos : usize,
    val : N
}

/// Verify if ix-dist and ix+dist (if they are valid indices) are likely peak
/// limits. Verify if the ratio slice[ix] / slice[ix-dist] AND slice[ix]/slice[ix+dist]
/// are both greater than rel.
pub fn verify_rel_high(slice : &[f32], ix : usize, dist : usize, rel : f32) -> bool {
    if slice.len() == 0 {
        return false;
    }
    if ix as i32 - (dist as i32) < 0 || (ix + dist) > (slice.len() - 1) {
        return false;
    }
    let val = slice[ix];
    let neighbor_left = slice[ix - dist];
    let neighbor_right = slice[ix + dist];
    val.abs() / neighbor_left.abs() > rel && val.abs() / neighbor_right.abs() > rel
}

/// Iterate over samples (assumed to be a time series) and find local peaks centered at (-win_len/2,+win_len/2).
/// (where win_len is assumed odd).
/// Peaks should have absolute value greater than magn, and optionally all ratios (peak - other) / delta_t
/// should have angle greater than slope (in radians of the normalized slop arc-tangent;
/// flat peaks approaching 0; sharp peaks approaching pi/4). The comparison is made between the
/// peak and all samples in its window.
/// The peaks vector is cleared and results are written into it. If col_stride is passed,
/// windows are assumed to be defined for an image in the samples slice, and the 2D neighborhood is searched
/// instead of the 1D neighborhood.
pub fn event_search(
    peaks : &mut Vec<(usize, f32)>,
    samples : &[f32],
    magn : f32,
    slope : Option<f32>,
    win_len : usize
) {
    peaks.clear();
    let half_len = win_len / 2;
    let n = samples.len();
    assert!(slope.map(|s| s >= 0.0 && s <= PI / 2.).unwrap_or(true));
    assert!(magn > 0.0);
    assert!(win_len % 2 != 0);

    for i in 0..samples.len() {
        if samples[i].abs() < magn {
            continue;
        }

        let mut highest = true;
        let mut is_steep = true;
        for j in (i.saturating_sub(half_len))..(i+half_len).min(n) {
        	if i == j {
        		continue;
        	}
            if (samples[j].abs() > samples[i].abs()) {
                highest = false;
                break;
            }

            // At this point, abs(samples[i]) > abs(samples[j]) always, allowing us to calculate a slope.
            if let Some(slope) = slope {
                // dx \in [0.0,1.0]; //dy \in [0.0, 1.0], which allow us to express the slope in radians
                let dx = (i as f32 - j as f32).abs() / (half_len as f32);
                let dy = (samples[i].abs() - samples[j].abs()) / (samples[i].abs() + f32::EPSILON);
                // println!("For i = {}", samples[i]);

                // println!("j = {}, dy = {}; dx = {}; dy/dx = {}; atan = {}", samples[j], dy, dx, dy/dx, (dy/dx).atan());
                if (dy / dx).atan() < slope {
                    is_steep = false;
                    break;
                }
            }
        }

        if highest && is_steep {
        	// println!("Added coef {} at {} to peak set", samples[i], i);
            peaks.push((i, samples[i]));
        }
    }

    if peaks.len() == 0 {
        return;
    }

    // Sort peaks from largest to smallest
    peaks.sort_by(|(_, magn_a), (_, magn_b)| magn_b.partial_cmp(&magn_a).unwrap_or(Ordering::Equal) );

    // Remove smaller peaks within same window
    let (mut i, mut j) = (0, 1);
    while i < peaks.len() - 1 {
        j = i + 1;

        // peaks[j] is always smaller than peaks[i] since the vector was sorted. Remove it if
        // it is within the window defined around peaks[i];
        while j < peaks.len() {
            if peaks[j].0 >= peaks[i].0.saturating_sub(half_len) && peaks[j].0 <= peaks[i].0 + half_len {
            	// println!("Removed coef {} at {} from peak set", samples[j], j);
                peaks.remove(j);
            } else {
                j += 1;
            }
        }
        i += 1;
    }

    // Sort peaks at their index order after peaks that are too close are removed.
    peaks.sort_by(|(pos_a, _), (pos_b, _)| pos_a.cmp(&pos_b) );
}

/// Search for indices (low, high) (not necessarily symmetric) around a center
/// index that guarantee the highest absolute difference between [low, high].
/// as long as the distance is within max_dist of the center for both sides.
pub fn longest_asymetric_diff<S>(signal : &[S], center : usize, max_dist : usize) -> Option<(usize, usize)>
where
    S : Scalar + std::ops::Sub<Output=S> + Real + Zero
{
    let mut low = 1;
    let mut high = 1;
    let mut highest_diff = S::zero();
    let mut found_any = false;
    for ix_low in 1..=max_dist {
        for ix_high in 1..=max_dist {
            if center as i32 - ix_low as i32 >= 0 && center + ix_high < signal.len() {
                let diff = (signal[center - ix_low] - signal[center + ix_high]).abs();
                if diff > highest_diff {
                    low = ix_low;
                    high = ix_high;
                    highest_diff = diff;
                    found_any = true;
                }
            }
        }
    }
    if found_any {
        Some((center - low, center + high))
    } else {
        None
    }
}

#[test]
fn peaks() {
    let mut peaks = Vec::new();
    let mut a = [0.0, 0.3, 0.6, 10.0, 16.0, 10.0, 0.6, 0.3, 0.0];
    // peak_finding(&mut peaks, &a[..], 5.0, Some(PI / 4.), 3);
    peak_finding(&mut peaks, &a[..], 5.0, None, 5);
    println!("Positive peak = {:?}", peaks);

    a.iter_mut().for_each(|a| *a = *a*(-1.0) );
    peak_finding(&mut peaks, &a[..], 5.0, None, 5);
    println!("Negative peak = {:?}", peaks);
}
