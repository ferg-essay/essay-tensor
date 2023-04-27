use std::marker::PhantomData;

use crate::{tensor::{Dtype, Uop, Binop}, Tensor};

pub trait Op {}
pub type BoxOp = Box<dyn Op>;

pub struct Expr<const N:usize, D:Dtype=f32> {
    op: Option<BoxOp>,
    tensor: Tensor<N, D>,
}

impl<const N:usize, D:Dtype> Expr<N, D> {
    pub fn uop(self, uop: impl Uop<D>) -> Self {
        todo!()
    }

    pub fn binop(self, op: impl Binop<D>, b: Self) -> Self {
        todo!()
    }
}