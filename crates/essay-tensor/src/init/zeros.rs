use crate::{Tensor, prelude::Shape};

pub fn zeros(shape: impl Into<Shape>) -> Tensor {
    Tensor::fill(0., shape)
}

impl Tensor {
    pub fn zeros(shape: impl Into<Shape>) -> Tensor {
        zeros(shape)
    }
}

