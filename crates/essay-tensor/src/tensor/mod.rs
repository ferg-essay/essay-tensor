mod data;
mod index;
mod slice;
mod shape;
mod stack;
mod tensor;
mod tensor_vec;

pub use data::{
    TensorUninit,
};

pub use shape::{
    Shape, AxisOpt, Axis,
    squeeze,
};

pub use stack::{
    stack,
};

pub use tensor::{
    Dtype, Tensor, TensorId, 
};

pub use tensor_vec::{
    TensorVec,
};
