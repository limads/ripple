use nalgebra::*;
use nalgebra::base::storage::Storage;
use std::convert::TryInto;
use super::*;

pub(crate) struct DWTIteratorBase<S> {
    max_lvl : usize,
    curr_lvl : usize,
    full : S
}

/// Calculates the DWT iterator maximum level based on a slice.
pub fn calc_max_level<N>(full : &[N]) -> usize {
    let n = full.len();
    assert!(super::is_valid_dwt_len(n));
    let max_lvl = ((n as f32).log2() - 1.) as usize;
    max_lvl
}

impl<'a, N> DWTIteratorBase<&'a Pyramid<N>> 
where
    N : Scalar + Copy
{
    pub fn new_ref(
        full : &'a Pyramid<N>
    ) -> DWTIteratorBase<&'a Pyramid<N>>
        where
            N : Scalar + Copy
    {
        let max_lvl = calc_max_level(full.as_ref());
        DWTIteratorBase::<&'a Pyramid<N>> {
            max_lvl,
            curr_lvl : 0,
            full
        }
    }
}

pub(crate) type DWTIterator1D<'a, N> = DWTIteratorBase<&'a Pyramid<N>>;

pub(crate) fn get_level_slice_1d<'a, N>(
    v : &'a DVector<N>,
    lvl : usize
) -> Option<DVectorSlice<'a, N>> 
where
    N : Scalar + Copy
{
    let lvl_pow = (2 as i32).pow(lvl.try_into().unwrap()) as usize;
    if lvl_pow > v.nrows() / 2 {
        return None;
    }
    let (lvl_off, lvl_sz) = match lvl {
        0 => (0, 2),
        l => (lvl_pow, lvl_pow)
    };
    Some(v.rows(lvl_off, lvl_sz))
}

impl<'a, N> Iterator for DWTIterator1D<'a, N> 
where
    N : Scalar + Copy
{

    type Item = DVectorSlice<'a, N>;

    fn next(&mut self) -> Option<DVectorSlice<'a, N>>
        where Self : 'a
    {
        let slice = get_level_slice_1d(&self.full.buf, self.curr_lvl)?;
        if self.curr_lvl == self.max_lvl + 1 {
            return None;
        }
        self.curr_lvl += 1;
        Some(slice)
    }

}

