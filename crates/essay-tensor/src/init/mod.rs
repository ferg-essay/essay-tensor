mod meshgrid;
mod arange;
mod initializer;
mod random_normal;
mod random_uniform;
mod linspace;
mod fill;
mod ones;
mod zeros;

pub use initializer::Initializer;

pub use ones::ones;
pub use fill::fill;

pub use arange::{
    arange,
};

pub use meshgrid::{
    meshgrid, meshgrid_ij, Meshgrid,
};

pub use zeros::{
    zeros, zeros_initializer,
};

pub use random_uniform::{
    random_uniform, random_uniform_initializer, UniformOpt
};

pub use random_normal::{
    random_normal, random_normal_initializer, NormalOpt
};

pub use linspace::linspace;