use crate::{ops::{Uop, uop}, tensor::Tensor};


enum Unary {
    ReLU,
    Softplus,
}

impl Uop for Unary {
    fn eval(&self, value: f32) -> f32 {
        match &self {
            Unary::ReLU => value.max(0.),
            Unary::Softplus => (value.exp() + 1.).ln(),
        }
    }
}

pub fn relu<const N:usize>(tensor: &Tensor<N>) -> Tensor<N> {
    uop(Unary::ReLU, tensor)
}

pub fn softplus<const N:usize>(tensor: &Tensor<N>) -> Tensor<N> {
    uop(Unary::Softplus, tensor)
}
