pub mod expr;
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

    pub use crate::tensor::{
        IntoTensor,
    };
}