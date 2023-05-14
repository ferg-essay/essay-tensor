use crate::{Tensor, prelude::Shape};

pub fn ones(shape: impl Into<Shape>) -> Tensor {
    Tensor::fill(1., shape)
}

impl Tensor {
    pub fn ones(shape: impl Into<Shape>) -> Tensor {
        Tensor::fill(1., shape)
    }
}

