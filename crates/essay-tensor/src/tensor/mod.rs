mod ops;
mod data;
mod tensor;

pub use data::{
    TensorData, TensorUninit,
};

pub use tensor::{
    Dtype, Tensor, Op, BoxOp, IntoTensor, NodeId,
};

pub use ops::{
    Uop, Binop, Fold, BiFold,
};
