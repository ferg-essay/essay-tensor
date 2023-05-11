mod fold_2;
mod binary_op;
mod fold;
mod unary_op;

pub use fold_2::{
    BiFold,
};

pub use fold::{
    fold_op, Fold,
};

pub use unary_op::{
    unary_op, Uop,
};

pub use binary_op::{
    binary_op, Binop,
};