mod data;
mod index;
mod slice;
mod shape;
mod tensor;
mod tensors;

pub use data::{
    TensorUninit,
};

pub use shape::{
    Shape,
};

pub use tensor::{
    Dtype, Tensor, TensorId,
};

pub use tensors::{
    Tensors,
};
