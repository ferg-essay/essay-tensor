use std::cmp;

use crate::{Tensor, tensor::{Dtype, TensorUninit}};

pub fn fill<D:Dtype>(fill: D, shape: &[usize]) -> Tensor<D> {
    unsafe {
        let len = cmp::max(1, shape.iter().product());

        let mut data = TensorUninit::<D>::new(len);

        for i in 0..len {
            data[i] = fill;
        }

        Tensor::new(data.init(), shape)
    }
}

impl<D:Dtype> Tensor<D> {
    pub fn fill(value: D, shape: &[usize]) -> Tensor<D> {
        fill(value, shape)
    }
}

#[cfg(test)]
mod test {
    use crate::{prelude::*, init::fill};
    
    #[test]
    fn fill_basic() {
        assert_eq!(fill(0., &[]), tensor!(0.));
        assert_eq!(fill(1., &[1]), tensor!([1.]));
        assert_eq!(fill(2., &[3]), tensor!([2., 2., 2.]));
        assert_eq!(fill(3., &[3, 2]), tensor!([[3., 3., 3.], [3., 3., 3.]]));
        assert_eq!(fill(4., &[1, 2, 3]), tensor!([
            [[4.], [4.]],
            [[4.], [4.]],
            [[4.], [4.]],
        ]));
    }
}