mod data;
mod index;
mod slice;
mod shape;
mod stack;
mod tensor;

pub use data::{
    TensorUninit,
};

pub use shape::{
    Shape, AxisOpt,
    squeeze,
};

pub use stack::{
    stack,
};

pub use tensor::{
    Dtype, Tensor, TensorId,
};
