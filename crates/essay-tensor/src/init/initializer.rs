use crate::{Tensor, prelude::Shape};

pub trait Initializer {
    fn init(&self, shape: &Shape) -> Tensor;
}