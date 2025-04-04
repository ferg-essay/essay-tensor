// mod math;
mod axis;
mod data;
mod index;
mod slice;
mod shape;
mod tensor;

#[cfg(test)]
mod test;

pub use axis::Axis;

pub(crate) use data::TensorData;

pub use shape::Shape;

pub use tensor::{
    Type, Tensor, IntoTensorList,
};
