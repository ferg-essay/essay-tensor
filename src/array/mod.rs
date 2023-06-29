mod axis;
mod flatten;
mod reshape;
mod stack;
mod squeeze;
mod transpose;

pub use axis::{
    Axis, AxisOpt,
};

pub use flatten::{
    flatten
};

pub use reshape::{
    reshape,
};

pub use stack::{
    stack,
};

pub use squeeze::{
    squeeze,
};

pub use transpose::{
    transpose,
};
