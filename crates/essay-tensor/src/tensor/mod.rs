mod data;
mod tensor;

pub use data::{
    TensorData, TensorUninit,
};

pub use tensor::{
    Dtype, Tensor, TensorId, IntoTensor, NodeId,
};
