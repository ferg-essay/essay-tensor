// mod math;
mod data;
mod index;
mod slice;
mod shape;
mod tensor;
mod tensor_vec;

pub(crate) use data::TensorData;

pub use shape::Shape;

pub use tensor::{
    Dtype, Tensor, IntoTensorList,
};
