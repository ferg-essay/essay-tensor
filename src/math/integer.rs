use num_traits::PrimInt;

use crate::tensor::{Tensor, Type};

impl<T: Type + PrimInt> Tensor<T> {
    #[inline]
    pub fn count_ones(&self) -> Tensor<u32> {
        self.map(|x| x.count_ones())
    }

    #[inline]
    pub fn count_zeros(&self) -> Tensor<u32> {
        self.map(|x| x.count_zeros())
    }

    #[inline]
    pub fn leading_ones(&self) -> Tensor<u32> {
        self.map(|x| x.leading_ones())
    }

    #[inline]
    pub fn leading_zeros(&self) -> Tensor<u32> {
        self.map(|x| x.leading_zeros())
    }

    #[inline]
    pub fn trailing_ones(&self) -> Tensor<u32> {
        self.map(|x| x.trailing_ones())
    }

    #[inline]
    pub fn trailing_zeros(&self) -> Tensor<u32> {
        self.map(|x| x.trailing_zeros())
    }

    #[inline]
    pub fn rotate_left(&self, b: &Tensor<u32>) -> Tensor<T> {
        self.map2(b, |x, n| x.rotate_left(n.clone()))
    }

    #[inline]
    pub fn rotate_right(&self, b: &Tensor<u32>) -> Tensor<T> {
        self.map2(b, |x, n| x.rotate_right(n.clone()))
    }

    #[inline]
    pub fn swap_bytes(&self) -> Tensor<T> {
        self.map(|x| x.swap_bytes())
    }

    #[inline]
    pub fn reverse_bits(&self) -> Tensor<T> {
        self.map(|x| x.reverse_bits())
    }

    #[inline]
    pub fn to_be(&self) -> Tensor<T> {
        self.map(|a| a.to_be())
    }

    #[inline]
    pub fn to_le(&self) -> Tensor<T> {
        self.map(|a| a.to_le())
    }
}