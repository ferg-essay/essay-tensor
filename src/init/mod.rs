//mod one_hot;
mod diag;
mod eye;
mod fill;
// mod geomspace;
mod initializer;
mod meshgrid;
mod ones;
mod random_normal;
mod random_uniform;
mod tri;
mod vector;
mod zeros;

pub use initializer::Initializer;

pub use fill::fill;

pub use diag::diagflat;

use num_traits::{One, Zero};
//pub use one_hot::one_hot;

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

pub use vector::{
    arange,
    linspace,
    logspace, logspace_opt,
    geomspace,
    one_hot,
};

pub use tri::{
    tri, tril, triu
};

use crate::tensor::{Shape, Tensor, Type};

pub fn ones<T: Type + One + Clone>(shape: impl Into<Shape>) -> Tensor<T> {
    Tensor::init(shape, || One::one())
}

pub fn zeros<T: Type + Zero + Clone>(shape: impl Into<Shape>) -> Tensor<T> {
    Tensor::init(shape, || Zero::zero())
}

impl<T: Type + Zero + Clone> Tensor<T> {
    #[inline]
    pub fn zeros(shape: impl Into<Shape>) -> Self {
        Self::init(shape, || Zero::zero())
    }
}

impl<T: Type + One + Clone> Tensor<T> {
    #[inline]
    pub fn ones(shape: impl Into<Shape>) -> Self {
        ones(shape)
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;
    use crate::init::zeros;

    #[test]
    fn test_zeros() {
        assert_eq!(zeros([1]), ten![0.]);
        assert_eq!(zeros([3]), ten![0., 0., 0.]);
        assert_eq!(zeros([2, 3]), ten![[0., 0., 0.], [0., 0., 0.]]);
    }

    #[test]
    fn test_zeros_types() {
        assert_eq!(zeros([1]), ten![0i8]);
        assert_eq!(zeros([1]), ten![0i16]);
        assert_eq!(zeros([1]), ten![0i32]);
        assert_eq!(zeros([1]), ten![0i64]);
        assert_eq!(zeros([1]), ten![0i128]);
        assert_eq!(zeros([1]), ten![0isize]);

        assert_eq!(zeros([1]), ten![0u8]);
        assert_eq!(zeros([1]), ten![0u16]);
        assert_eq!(zeros([1]), ten![0u32]);
        assert_eq!(zeros([1]), ten![0u64]);
        assert_eq!(zeros([1]), ten![0u128]);
        assert_eq!(zeros([1]), ten![0usize]);

        assert_eq!(zeros([1]), ten![0.0f32]);
        assert_eq!(zeros([1]), ten![0.0f64]);
    }
}

