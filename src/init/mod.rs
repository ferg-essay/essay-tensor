mod arange;
mod diag;
mod eye;
mod fill;
mod geomspace;
mod initializer;
mod linspace;
mod logspace;
mod meshgrid;
mod ones;
mod random_normal;
mod random_uniform;
mod tri;
mod zeros;

pub use initializer::Initializer;

pub use ones::ones;
pub use fill::fill;

pub use arange::{
    arange,
};

pub use diag::{
    diagflat,
};

pub use eye::{
    eye, identity,
};

pub use meshgrid::{
    meshgrid, meshgrid_ij, Meshgrid, mgrid,
};

pub use random_uniform::{
    random_uniform, random_uniform_initializer, UniformOpt
};

pub use random_normal::{
    random_normal, random_normal_initializer, NormalOpt
};

pub use linspace::linspace;
pub use logspace::{
    logspace, logspace_opt
};
pub use geomspace::{
    geomspace,
};

pub use tri::{
    tri, tril, triu
};

pub use zeros::{
    zeros, zeros_initializer,
};
