use crate::{Tensor, tensor::{Dtype, TensorUninit}};

pub fn fill<D:Dtype>(fill: D, shape: &[usize]) -> Tensor<D> {
    unsafe {
        let len = shape.iter().product();
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
