mod slice;
mod index;
mod data;
mod tensor;
mod tensors;

pub use data::{
    TensorUninit,
};

pub use tensors::{
    Tensors,
};

pub use tensor::{
    Dtype, Tensor, TensorId, Shape,
};
