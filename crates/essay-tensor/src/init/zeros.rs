use crate::{Tensor, tensor::{Dtype, TensorUninit}};

pub fn zeros(shape: &[usize]) -> Tensor {
    Tensor::fill(0., shape)
}

impl Tensor {
    pub fn zeros(shape: &[usize]) -> Tensor {
        zeros(shape)
    }
}
