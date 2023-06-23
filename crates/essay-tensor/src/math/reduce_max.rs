use crate::{Tensor, ops::{reduce_op, ReduceKernel, ReduceOpt}};

#[derive(Debug, Copy, Clone)]
pub struct ReduceMax;

pub fn reduce_max(a: &Tensor) -> Tensor {
    reduce_op(a, ReduceMax, ())
}

pub fn reduce_max_opt(a: &Tensor, opt: impl ReduceOpt) -> Tensor {
    reduce_op(a, ReduceMax, opt)
}

impl Tensor {
    pub fn reduce_max(&self) -> Tensor {
        reduce_max(self)
    }

    pub fn reduce_max_opt(&self, opt: impl ReduceOpt) -> Tensor {
        reduce_max_opt(self, opt)
    }
}

impl ReduceKernel<f32> for ReduceMax {
    #[inline]
    fn init(&self) -> f32 {
        f32::MIN
    }
    
    #[inline]
    fn f(&self, acc: f32, a: f32) -> f32 {
        acc.max(a)
    }

    #[inline]
    fn df_dx(&self, a: f32) -> f32 {
        todo!()
    }
}
