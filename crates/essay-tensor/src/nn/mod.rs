mod softmax;
mod l2_loss;

use crate::{tensor::{Tensor}, 
    tensor_uop, 
    ops::{Uop, unary_op}
};

pub use l2_loss::l2_loss;


#[derive(Debug, Copy, Clone, PartialEq)]
enum Unary {
    ReLU,
    Softplus,
}

impl Uop<f32> for Unary {
    fn f(&self, value: f32) -> f32 {
        match &self {
            Unary::ReLU => value.max(0.),
            Unary::Softplus => (value.exp() + 1.).ln(),
        }
    }

    fn df_dx(&self, _value: f32) -> f32 {
        todo!()
    }
}

tensor_uop!(relu, Unary::ReLU);
tensor_uop!(softplus, Unary::Softplus);
