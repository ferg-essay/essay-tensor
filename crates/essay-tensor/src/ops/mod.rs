mod binary_op;
mod fold;
mod fold_2;
mod init_op;
mod unary_op;
pub mod reduce;

pub use unary_op::{
    unary_op, UnaryKernel,
};

pub use init_op::{
    init_op, InitKernel,
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
