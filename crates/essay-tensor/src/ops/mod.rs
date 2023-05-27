mod unary_op;
mod binary_op;
pub mod reduce;
mod fold;
mod fold_2;

pub use unary_op::{
    unary_op, UnaryKernel,
};

pub use binary_op::{
    binary_op, BinaryKernel,
};

pub use reduce::{
    reduce_op, ReduceKernel, ReduceOpt,
};

pub use fold_2::{
    BiFold,
};

pub use fold::{
    fold_op, Fold,
};
