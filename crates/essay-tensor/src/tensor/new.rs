use crate::Tensor;

use super::{TensorUninit, Dtype};


impl<D:Dtype> Tensor<D> {
    pub fn fill(fill: D, shape: &[usize]) -> Tensor<D> {
        unsafe {
            let len = shape.iter().product();
            let mut data = TensorUninit::<D>::new(len);

            for i in 0..len {
                data[i] = fill;
            }

            Tensor::new(data.init().into(), shape)
        }
    }
}

impl Tensor {
    pub fn zeros(shape: &[usize]) -> Tensor {
        Tensor::fill(0., shape)
    }

    pub fn ones(shape: &[usize]) -> Tensor {
        Tensor::fill(1., shape)
    }
}

