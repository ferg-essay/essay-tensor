// mod math;
mod data;
mod index;
mod slice;
mod shape;
mod tensor;
mod tensor_vec;

pub(crate) use data::{TensorData, TensorUninit};

pub use shape::Shape;

pub use tensor::{
    Dtype, Tensor, IntoTensorList, C32, C64
};

pub use tensor_vec::TensorVec;
