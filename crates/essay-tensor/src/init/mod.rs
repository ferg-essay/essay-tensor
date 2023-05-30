mod random_normal;
mod random_uniform;
mod linspace;
mod fill;
mod ones;
mod zeros;

pub use ones::ones;
pub use fill::fill;
pub use zeros::zeros;

pub use random_uniform::{
    random_uniform, UniformOpt
};

pub use random_normal::{
    random_normal, NormalOpt
};

pub use linspace::linspace;