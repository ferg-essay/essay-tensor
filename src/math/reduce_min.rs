use crate::{Tensor, ops::{reduce_op, ReduceKernel, ReduceOpt}};

#[derive(Debug, Copy, Clone)]
pub struct ReduceMin;

pub fn reduce_min(a: &Tensor) -> Tensor {
    reduce_op(a, ReduceMin, ())
}

pub fn reduce_min_opt(a: &Tensor, opt: impl ReduceOpt) -> Tensor {
    reduce_op(a, ReduceMin, opt)
}

impl Tensor {
    /*
    pub fn reduce_min(&self) -> Tensor {
        reduce_min(self)
    }
    */

    pub fn reduce_min_opt(&self, opt: impl ReduceOpt) -> Tensor {
        reduce_min_opt(self, opt)
    }
}

impl ReduceKernel<f32> for ReduceMin {
    #[inline]
    fn init(&self) -> f32 {
        f32::MAX
    }
    
    #[inline]
    fn f(&self, acc: f32, a: f32) -> f32 {
        acc.min(a)
    }

    #[inline]
    fn df_dx(&self, _x: f32) -> f32 {
        todo!()
    }
}
