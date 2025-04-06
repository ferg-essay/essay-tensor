//mod one_hot;
mod diag;
mod eye;
// mod geomspace;
mod initializer;
mod meshgrid;
mod random;
mod tri;
mod vector;

pub use initializer::Initializer;

pub use diag::diagflat;

use num_traits::{One, Zero};
//pub use one_hot::one_hot;

pub use eye::{
    eye, identity,
};

pub use meshgrid::{
    meshgrid, meshgrid_ij, Meshgrid, mgrid,
};

pub use random::{
    random_normal, Normal,
    random_uniform, Uniform
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

pub fn fill<T: Type + Clone>(shape: impl Into<Shape>, fill: T) -> Tensor<T> {
    Tensor::init(shape, || fill.clone())
}

impl<T: Type + Zero + Clone> Tensor<T> {
    #[inline]
    pub fn zeros(shape: impl Into<Shape>) -> Self {
        zeros(shape)
    }
}

impl<T: Type + One + Clone> Tensor<T> {
    #[inline]
    pub fn ones(shape: impl Into<Shape>) -> Self {
        ones(shape)
    }
}

impl<T: Type + Clone> Tensor<T> {
    #[inline]
    pub fn fill(shape: impl Into<Shape>, value: T) -> Self {
        fill(shape, value)
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;
    use crate::init::{fill, zeros};

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

    #[test]
    fn fill_basic() {
        assert_eq!(fill([], 0.), Tensor::from(0.));
        assert_eq!(fill([1], 1.), ten![1.]);
        assert_eq!(fill([3], 2.), ten![2., 2., 2.]);
        assert_eq!(fill([2, 3], 3.), ten![[3., 3., 3.], [3., 3., 3.]]);
        assert_eq!(fill([3, 2, 1], 4.), ten![
            [[4.], [4.]],
            [[4.], [4.]],
            [[4.], [4.]],
        ]);
    }
}

