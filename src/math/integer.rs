use num_traits::PrimInt;

use crate::tensor::{Tensor, Type};

macro_rules! map {
    ($id: ident) => {
        #[inline]
        pub fn $id(&self) -> Tensor<T> {
            self.map(|a| a.$id())
        }
    }
}

macro_rules! map_to_u32 {
    ($id: ident) => {
        #[inline]
        pub fn $id(&self) -> Tensor<u32> {
            self.map(|a| a.$id())
        }
    }
}

impl<T: Type + PrimInt> Tensor<T> {
    map_to_u32!(count_ones);
    map_to_u32!(count_zeros);
    map_to_u32!(leading_ones);
    map_to_u32!(leading_zeros);
    map_to_u32!(trailing_ones);
    map_to_u32!(trailing_zeros);

    #[inline]
    pub fn rotate_left(&self, b: &Tensor<u32>) -> Tensor<T> {
        self.map2(b, |x, n| x.rotate_left(n.clone()))
    }

    #[inline]
    pub fn rotate_right(&self, b: &Tensor<u32>) -> Tensor<T> {
        self.map2(b, |x, n| x.rotate_right(n.clone()))
    }

    map!(swap_bytes);
    map!(reverse_bits);
    map!(to_be);
    map!(to_le);
}