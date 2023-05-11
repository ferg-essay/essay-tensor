pub mod activation;
pub mod init;
pub mod linalg;
pub mod loss;
pub mod nn;
pub mod graph;
pub mod math;
pub mod ops;
pub mod random;
pub mod tensor;

#[macro_use]
pub mod macros;

pub use tensor::Tensor;

pub mod prelude {
    pub use crate::tensor;
    pub use crate::Tensor;
    pub use crate::graph::{
        Trainer, Bundle
    };
}