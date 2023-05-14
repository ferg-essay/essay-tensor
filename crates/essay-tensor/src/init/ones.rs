use crate::{Tensor, prelude::IntoShape};

pub fn ones(shape: impl IntoShape) -> Tensor {
    Tensor::fill(1., shape)
}

impl Tensor {
    pub fn ones(shape: impl IntoShape) -> Tensor {
        Tensor::fill(1., shape)
    }
}

