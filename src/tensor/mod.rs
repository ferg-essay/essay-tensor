mod data;
mod index;
mod slice;
mod shape;
mod concat;
mod tensor;
mod tensor_vec;

pub use data::{
    TensorUninit,
};

pub use shape::{
    Shape, AxisOpt, Axis,
    squeeze,
};

pub use concat::{
    stack,
};

pub use tensor::{
    Dtype, Tensor, TensorId, C32, C64
};

pub use tensor_vec::{
    TensorVec,
};
