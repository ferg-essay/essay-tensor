use crate::tensor::{Shape, Tensor};

pub trait Initializer {
    fn init(&self, shape: &Shape) -> Tensor;
}