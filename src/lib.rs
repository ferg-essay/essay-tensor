pub mod array;
pub mod signal;
pub mod io;
// pub mod flow;
// pub mod dataset;
pub mod init;
pub mod linalg;
pub mod math;
//pub mod ops;
pub mod random;
pub mod stats;
pub mod tensor;
#[cfg(test)]
pub(crate) mod test;

#[macro_use]
pub mod macros;

// pub use tensor::Tensor;

pub mod prelude {
    pub use crate::{ten, tf32};
    pub use crate::tensor::Shape;
    pub use crate::tensor::Axis;
    pub use crate::tensor::Tensor;
    // pub use crate::dataset::Dataset;
    //pub use crate::ops::ReduceOpt;
}