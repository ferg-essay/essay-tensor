use crate::{Tensor, ops::{reduce_op, ReduceKernel, ReduceOpt}};

#[derive(Debug, Copy, Clone)]
pub struct ReduceSum;

pub fn reduce_sum(a: &Tensor) -> Tensor {
    reduce_op(a, ReduceSum, ())
}

pub fn reduce_sum_opt(a: &Tensor, opt: impl ReduceOpt) -> Tensor {
    reduce_op(a, ReduceSum, opt)
}

impl Tensor {
    pub fn reduce_sum(&self) -> Tensor {
        reduce_sum(self)
    }

    pub fn reduce_sum_opt(&self, opt: impl ReduceOpt) -> Tensor {
        reduce_sum_opt(self, opt)
    }
}

impl ReduceKernel<f32> for ReduceSum {
    #[inline]
    fn init(&self) -> f32 {
        0.
    }
    
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
    use crate::{prelude::*, ops::ReduceOpt};

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

    #[test]
    fn reduce_sum_1xn_axis_none() {
        assert_eq!(tf32!([[1.]]).reduce_sum_opt(().axis(None)), tf32!(1.));
        assert_eq!(tf32!([[1., 10.]]).reduce_sum_opt(().axis(None)), tf32!(11.));
        assert_eq!(tf32!([[10., 1.]]).reduce_sum_opt(().axis(None)), tf32!(11.));
    }

    #[test]
    fn reduce_sum_2xn_axis_none() {
        assert_eq!(tf32!([[1.], [2.]]).reduce_sum_opt(().axis(None)), tf32!(3.));
        assert_eq!(tf32!([[1., 10.], [100., 1000.]]).reduce_sum_opt(().axis(None)), tf32!(1111.));
    }

    #[test]
    fn reduce_sum_2x1x1xn_axis_none() {
        assert_eq!(tf32!([[[[1.]]], [[[2.]]]]).reduce_sum_opt(().axis(None)), tf32!(3.));
        assert_eq!(tf32!([[[[1., 10.]]], [[[100., 1000.]]]]).reduce_sum_opt(().axis(None)), tf32!(1111.));
    }

    #[test]
    fn reduce_sum_2xn_axis_0() {
        assert_eq!(tf32!([[1.], [2.]]).reduce_sum_opt(().axis(Some(0))), tf32!([3.]));
        assert_eq!(tf32!([[1., 10.], [100., 1000.]]).reduce_sum_opt(().axis(Some(0))), tf32!([101., 1010.]));
    }
}
