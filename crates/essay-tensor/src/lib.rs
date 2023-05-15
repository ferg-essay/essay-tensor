pub mod flow;
pub mod data;
pub mod model;
pub mod layer;
pub mod activation;
pub mod init;
pub mod linalg;
pub mod loss;
pub mod nn;
pub mod eval;
pub mod math;
pub mod ops;
pub mod random;
pub mod tensor;

#[macro_use]
pub mod macros;

pub use tensor::Tensor;

pub mod prelude {
    pub use crate::{tensor, tf32};
    pub use crate::tensor::{
        Bundle, Shape,
    };
    pub use crate::Tensor;
    pub use crate::eval::{
        Trainer, 
    };
    pub use crate::data::{
        Dataset,
    };
}