use crate::tensor::{Type, Tensor, unsafe_init};

impl<D: Type + Clone> Tensor<D> {
    pub fn tile(
        &self, 
        multiples: impl Into<Tensor<usize>>, 
    ) -> Tensor<D> {
        tile(self, multiples)
    }
}

pub fn tile<T>(tensor: impl Into<Tensor<T>>, multiples: impl Into<Tensor<usize>>) -> Tensor<T>
where
    T: Type + Clone
{
    let tensor = tensor.into();

    let shape = tensor.shape();

    let mut shape_r = Vec::from(shape.as_vec());
    shape_r.reverse();

    let multiples = multiples.into();
    let mut mult_r = Vec::from(multiples.as_slice());
    mult_r.reverse();

    let n_inner = shape.size();
    let n_outer : usize = multiples.iter().product();
    let slice = tensor.as_slice();

    let mut o_shape_r = Vec::<usize>::new();
    let rank = shape.rank();
    let m_len = multiples.len();
    let o_rank = rank.max(multiples.len());
    for i in 0..o_rank {
        let tile_dim = if i < rank { shape_r[i] } else { 1 };
        let repeat_dim = if i < m_len { mult_r[i] } else { 1 };

        o_shape_r.push(tile_dim * repeat_dim);
    }
    o_shape_r.reverse();
    let o_shape = o_shape_r;

    unsafe {
        unsafe_init::<T>(n_outer * n_inner, o_shape, |o| {
            let mut offset = 0;
            tile_rec(
                o,
                slice, 
                0,
                slice.len(),
                &shape_r,
                &mult_r,
                o_rank - 1, 
                &mut offset
            )
        })
    }
}

// TODO: simplify this logic, possibly by reversing the shapes before
// starting
unsafe fn tile_rec<D: Type + Clone>(
    o: *mut D, 
    x: &[D],
    x_off: usize,
    x_len: usize,
    x_shape_r: &[usize],
    mult_r: &[usize],
    j: usize,
    offset: &mut usize
) {
    let x_rank = x_shape_r.len();
    let m_rank = mult_r.len();

    let repeat = if j < m_rank { mult_r[j] } else { 1 };

    for _ in 0..repeat {
        if j == 0 {
            // TODO: lookup ptr::copy_non_overlapping
            for i in 0..x_len {
                o.add(*offset + i)
                    .write(x[x_off + i].clone());
            }
            *offset += x_len;
        } else if j < x_rank {
            let x_len = x_shape_r.iter().take(j).product();

            for i in 0..x_shape_r[j] {
                tile_rec(o, x, x_off + i * x_len, x_len, x_shape_r, mult_r, j - 1, offset);
            }
        } else {
            tile_rec(o, x, x_off, x_len, x_shape_r, mult_r, j - 1, offset)
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{array::tile, ten};
    
    #[test]
    fn test_tile() {
        assert_eq!(tile(
            ten!([1.]),
            [3]
        ), ten!([1., 1., 1.]));

        assert_eq!(tile(
            ten!([1.]),
            [2, 3]
        ), ten!([[1., 1., 1.], [1., 1., 1.]]));

        assert_eq!(tile(
            ten!([1., 2.]),
            [3]
        ), ten!([1., 2., 1., 2., 1., 2.]));

        assert_eq!(tile(
            ten!([[1.], [2.]]),
            [3]
        ), ten!([[1., 1., 1.], [2., 2., 2.]]));

        assert_eq!(tile(
            ten!([1., 2.]),
            [2, 3]
        ), ten!([
            [1., 2., 1., 2., 1., 2.], 
            [1., 2., 1., 2., 1., 2.], 
        ]));
    }

    #[test]
    fn tile_i32() {
        assert_eq!(tile(
            ten![1, 2],
            [2, 3]
        ), ten![
            [1, 2, 1, 2, 1, 2], 
            [1, 2, 1, 2, 1, 2], 
        ]);
    }
}
