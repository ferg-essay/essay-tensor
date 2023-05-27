use crate::{Tensor, ops::{reduce_op, ReduceKernel}};

#[derive(Debug, Copy, Clone)]
pub struct ReduceSum;

pub fn reduce_sum(a: &Tensor) -> Tensor {
    reduce_op(a, ReduceSum, Some(-1))
}

pub fn reduce_sum_axis(a: &Tensor, axis: Option<i32>) -> Tensor {
    if let Some(axis) = axis {
        assert!(axis < a.rank() as i32);
        assert!(- axis < a.rank() as i32);
    }

    reduce_op(a, ReduceSum, axis)
}

impl Tensor {
    pub fn reduce_sum(&self) -> Tensor {
        reduce_sum(self)
    }

    pub fn reduce_sum_axis(&self, axis: Option<i32>) -> Tensor {
        reduce_sum_axis(self, axis)
    }
}

impl ReduceKernel for ReduceSum {
    #[inline]
    fn f(&self, acc: f32, a: f32) -> f32 {
        acc + a
    }

    #[inline]
    fn df_dx(&self, a: f32) -> f32 {
        a
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn reduce_sum_n() {
        assert_eq!(tf32!([1.]).reduce_sum(), tf32!(1.));
        assert_eq!(tf32!([1., 10.]).reduce_sum(), tf32!(11.));
        assert_eq!(tf32!([10., 1.]).reduce_sum(), tf32!(11.));
    }

    #[test]
    fn reduce_sum_1xn() {
        assert_eq!(tf32!([[1.]]).reduce_sum(), tf32!([1.]));
        assert_eq!(tf32!([[1., 10.]]).reduce_sum(), tf32!([11.]));
        assert_eq!(tf32!([[10., 1.]]).reduce_sum(), tf32!([11.]));
    }

    #[test]
    fn reduce_sum_2xn() {
        assert_eq!(tf32!([[1.], [2.]]).reduce_sum(), tf32!([1., 2.]));
        assert_eq!(tf32!([[1., 10.], [2., 20.]]).reduce_sum(), tf32!([11., 22.]));
        assert_eq!(tf32!([[20., 2.], [10., 1.]]).reduce_sum(), tf32!([22., 11.]));
    }

    #[test]
    fn reduce_sum_2x1xn() {
        assert_eq!(tf32!([[[1.]], [[2.]]]).reduce_sum(), tf32!([[1.], [2.]]));
        assert_eq!(tf32!([[[1., 10.]], [[2., 20.]]]).reduce_sum(), tf32!([[11.], [22.]]));
        assert_eq!(tf32!([[[20., 2.]], [[10., 1.]]]).reduce_sum(), tf32!([[22.], [11.]]));
    }
}
