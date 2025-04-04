// mod math;
mod axis;
mod data;
mod index;
mod slice;
mod shape;
mod tensor;

pub use axis::Axis;

pub(crate) use data::TensorData;

pub use shape::Shape;

pub use tensor::{
    Dtype, Tensor, IntoTensorList,
};
