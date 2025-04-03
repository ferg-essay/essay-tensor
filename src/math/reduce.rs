use std::ops;

use num_traits::Zero;

use crate::Tensor;


pub fn reduce_mean(a: &Tensor, opt: impl ReduceOpt) -> Tensor {
    // reduce_op(a, ReduceMean, opt)
    todo!();
}

impl Tensor {
    pub fn reduce_mean(&self, opt: impl ReduceOpt) -> Tensor {
        // reduce_mean(self, opt)
        todo!();
    }

    pub fn reduce_std(&self, opt: impl ReduceOpt) -> Tensor {
        // reduce_std(self, opt)
        todo!();
    }
    
    pub fn reduce_var(&self, opt: impl ReduceOpt) -> Tensor {
        // reduce_std(self, opt)
        todo!();
    }

    pub fn reduce_min(&self) -> Tensor {
        // reduce_min(self)
        todo!();
    }

    pub fn reduce_min_opt(&self, opt: impl ReduceOpt) -> Tensor {
        // reduce_min_opt(self, opt)
        todo!();
    }

    pub fn reduce_max(&self) -> Tensor {
        // self.fold(T::max_value(), |s, v| s.min(v))
        todo!()
    }

    pub fn reduce_max_opt(&self, opt: impl ReduceOpt) -> Tensor {
        // reduce_min_opt(self, opt)
        todo!();
    }
}

impl<T: ops::Add + Zero + Clone + 'static> Tensor<T> {
    pub fn reduce_sum(&self) -> Tensor<T> {
        //self.fold(T::zero(), |s, v| s + v)
        todo!()
    }

    pub fn reduce_product(&self) -> Tensor<T> {
        //self.fold(T::one(), |s, v| s * v)
        todo!()
    }
}

pub trait ReduceOpt {
    fn axis(self, axis: Option<i32>) -> ReduceArg;
    fn into(self) -> ReduceArg;
}

#[derive(Default, Clone, Debug, PartialEq)]
pub struct ReduceArg {
    axis: Option<i32>,
}

impl ReduceOpt for ReduceArg {
    fn axis(self, axis: Option<i32>) -> ReduceArg {
        Self { axis, ..self }
    }

    fn into(self) -> ReduceArg {
        self
    }
}

impl ReduceOpt for () {
    fn axis(self, axis: Option<i32>) -> ReduceArg {
        ReduceArg::default().axis(axis)
    }

    fn into(self) -> ReduceArg {
        ReduceArg::default()
    }
}

impl ReduceOpt for i32 {
    fn axis(self, axis: Option<i32>) -> ReduceArg {
        ReduceArg::default().axis(axis)
    }

    fn into(self) -> ReduceArg {
        ReduceArg::default().axis(Some(self))
    }
}
