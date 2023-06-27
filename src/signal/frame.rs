use crate::{Tensor, tensor::TensorId};

pub fn frame(x: impl Into<Tensor>, len: usize, step: usize) -> Tensor {
    let x = x.into();
    let pad_end = false;

    let n_frames = if pad_end {
        - (- (x.dim(0) as isize) / step as isize)
    } else {
        1 + (x.dim(0) as isize - len as isize) / step as isize
    } as usize;

    let mut vec = Vec::<Tensor>::new();
    for i in 0..n_frames {
        vec.push(x.subslice(i * step, len))
    }

    let mut shape = Vec::from(x.shape().as_slice());
    shape[0] = len;
    shape.insert(0, n_frames);

    Tensor::from_merge(vec, shape, TensorId::NONE)
}

#[cfg(test)]
mod test {
    use crate::{prelude::*, signal::frame};

    #[test]
    fn test_frame_2_2() {
        let x = tf32!([1., 2., 3., 4., 5., 6.]);
        assert_eq!(frame(&x, 2, 2), tf32!([[1., 2.], [3., 4.], [5., 6.]]));

        let x = tf32!([1., 2., 3., 4., 5., 6.]);
        assert_eq!(frame(&x, 3, 3), tf32!([[1., 2., 3.], [4., 5., 6.]]));
    }

    #[test]
    fn test_frame_3_2() {
        let x = tf32!([1., 2., 3., 4., 5., 6.]);
        assert_eq!(frame(&x, 3, 2), tf32!([[1., 2., 3.], [3., 4., 5.]]));
    }
}