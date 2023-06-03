pub mod signal;
pub mod io;
mod optimizer;
pub mod flow;
pub mod dataset;
pub mod layer;
pub mod activation;
pub mod init;
pub mod linalg;
pub mod loss;
pub mod nn;
pub mod model;
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
        Shape, AxisOpt
    };
    pub use crate::Tensor;
    pub use crate::model::{
        Trainer, 
    };
    pub use crate::dataset::{
        Dataset,
    };
    pub use crate::ops::{
        ReduceOpt,
    };
}