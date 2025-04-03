mod axis;
mod concat;
mod expand_dims;
mod flatten;
mod reshape;
mod hstack;
mod dstack;
mod split;
mod stack;
mod squeeze;
mod tile;
mod transpose;
mod vstack;

pub use axis::{
    Axis, AxisOpt,
};

pub use concat::{
    // concatenate,
};

pub use dstack::{
    dstack,
};

pub use expand_dims::{
    expand_dims
};

pub use flatten::{
    flatten
};

pub use hstack::{
    hstack,
};

pub use reshape::{
    reshape,
};

pub use split::{
    split, vsplit, hsplit, dsplit,
};

pub use stack::{
    stack,
};

pub use squeeze::{
    squeeze,
};

pub use tile::{
    tile,
};

pub use transpose::{
    transpose,
};

pub use vstack::{
    vstack,
};

use crate::{tensor::IntoTensorList, Tensor};

pub use concat::{concatenate, concatenate_axis};