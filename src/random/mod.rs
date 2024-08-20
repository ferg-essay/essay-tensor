mod random;
mod uniform;

pub use uniform::{
    uniform, uniform_b,
};

pub use random::{Rand32, Rand64, random_seed};
