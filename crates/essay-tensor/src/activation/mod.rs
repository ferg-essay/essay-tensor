mod relu;
mod softmax;
mod softplus;

use crate::{tensor_uop, Tensor, ops::unary_op};

pub use softmax::softmax;
//pub use softmax::softplusmax;

tensor_uop!(relu, relu::ReLU);
tensor_uop!(softplus, softplus::SoftPlus);
