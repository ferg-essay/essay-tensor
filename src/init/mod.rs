mod matrix;
mod initializer;
//mod meshgrid;
mod random;
mod vector;

pub use initializer::Initializer;

use num_traits::{One, Zero};

pub use matrix::{
    eye, identity,
    meshgrid, meshgrid_ij, Meshgrid, mgrid,
    tri,
};

//pub use meshgrid::{
//    meshgrid, meshgrid_ij, Meshgrid, mgrid,
//};

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

use crate::tensor::{Shape, Tensor, Type};

pub fn ones<T: Type + One>(shape: impl Into<Shape>) -> Tensor<T> {
    Tensor::init(shape, || One::one())
}

pub fn zeros<T: Type + Zero>(shape: impl Into<Shape>) -> Tensor<T> {
    Tensor::init(shape, || Zero::zero())
}

pub fn fill<T: Type + Clone>(shape: impl Into<Shape>, fill: T) -> Tensor<T> {
    Tensor::init(shape, || fill.clone())
}

impl<T: Type + Zero> Tensor<T> {
    #[inline]
    pub fn zeros(shape: impl Into<Shape>) -> Self {
        zeros(shape)
    }
}

impl<T: Type + One> Tensor<T> {
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
    use crate::init::{fill, ones, zeros};
    use crate::test::{C, O, Z};

    #[test]
    fn test_zeros() {
        assert_eq!(zeros([1]), ten![0.]);
        assert_eq!(zeros([3]), ten![0., 0., 0.]);
        assert_eq!(zeros([2, 3]), ten![[0., 0., 0.], [0., 0., 0.]]);

        assert_eq!(zeros([2]), ten![Z(0), Z(0)]);
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
    fn test_ones() {
        assert_eq!(ones([1]), ten![1.]);
        assert_eq!(ones([3]), ten![1, 1, 1]);
        assert_eq!(ones([2, 3]), ten![[1, 1, 1], [1, 1, 1]]);

        assert_eq!(ones([2]), ten![O(1), O(1)]);
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

        assert_eq!(fill([3], C(2)), ten![C(2), C(2), C(2)]);
    }
}

