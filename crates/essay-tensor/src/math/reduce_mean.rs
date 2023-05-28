use crate::{Tensor, ops::{reduce_op, ReduceKernel, ReduceOpt, reduce}};

#[derive(Debug, Copy, Clone)]
pub struct ReduceMean;

pub fn reduce_mean(a: &Tensor, opt: impl ReduceOpt) -> Tensor {
    reduce_op(a, ReduceMean, opt)
}

impl Tensor {
    pub fn reduce_mean(&self, opt: impl ReduceOpt) -> Tensor {
        reduce_mean(self, opt)
    }
}

impl ReduceKernel<Acc> for ReduceMean {
    #[inline]
    fn f(&self, state: Acc, a: f32) -> Acc {
        Acc {
            n: state.n + 1,
            sum: state.sum + a,
        }
    }

    #[inline]
    fn df_dx(&self, a: f32) -> f32 {
        a
    }
}

struct Acc {
    n: usize,
    sum: f32,
}

impl reduce::State for Acc {
    type Value = f32;

    fn value(&self) -> Self::Value {
        if self.n > 0 { self.sum / (self.n as f32) } else { 0. }
    }
}

impl Default for Acc {
    fn default() -> Self {
        Self { 
            n: Default::default(), 
            sum: Default::default() 
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{prelude::*};

    #[test]
    fn reduce_mean() {
        assert_eq!(tf32!([1.]).reduce_mean(()), tf32!(1.));
        assert_eq!(tf32!([1., 3.]).reduce_mean(()), tf32!(2.));
        assert_eq!(tf32!([[1., 3.], [4., 0.]]).reduce_mean(()), tf32!(2.));
        assert_eq!(tf32!([[[1., 3.]], [[4., 6.]]]).reduce_mean(().axis(Some(-1))), tf32!(3.5));
    }

    #[test]
    fn reduce_mean_axis_m1() {
        assert_eq!(tf32!([1.]).reduce_mean(().axis(Some(-1))), tf32!(1.));
        assert_eq!(tf32!([1., 3.]).reduce_mean(().axis(Some(-1))), tf32!(2.));
        assert_eq!(tf32!([[1., 3.], [4., 6.]]).reduce_mean(().axis(Some(-1))), tf32!([2., 5.]));
        assert_eq!(tf32!([[[1., 3.]], [[4., 6.]]]).reduce_mean(().axis(Some(-1))), tf32!([[2.], [5.]]));
    }

    #[test]
    fn reduce_mean_axis_0() {
        assert_eq!(tf32!([1.]).reduce_mean(().axis(Some(0))), tf32!(1.));
        assert_eq!(tf32!([1., 3.]).reduce_mean(().axis(Some(0))), tf32!(2.));
        assert_eq!(tf32!([[1., 3.], [4., 6.]]).reduce_mean(().axis(Some(0))), tf32!([2.5, 4.5]));
        assert_eq!(tf32!([[[1., 3.]], [[4., 6.]]]).reduce_mean(().axis(Some(0))), tf32!([[2.5, 4.5]]));
    }
}