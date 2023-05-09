pub mod ops;
pub mod module;
pub mod random;
pub mod linalg;
pub mod nn;
pub mod math;
pub mod tensor;

#[macro_use]
pub mod macros;

pub use tensor::Tensor;

pub mod prelude {
    pub use crate::tensor;
    pub use crate::Tensor;
    pub use crate::module::{
        Module, Bundle
    };
}