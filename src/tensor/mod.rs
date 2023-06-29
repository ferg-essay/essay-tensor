mod data;
mod index;
mod slice;
mod shape;
mod tensor;
mod tensor_vec;

pub use data::{
    TensorUninit,
};

pub use shape::{
    Shape, 
};

pub use tensor::{
    Dtype, Tensor, TensorId, IntoTensorList, C32, C64
};

pub use tensor_vec::{
    TensorVec,
};
