use crate::{Tensor, ops::{reduce_op, ReduceKernel, ReduceOpt, reduce}};

#[derive(Debug, Copy, Clone)]
pub struct ReduceVariance;

pub fn reduce_variance(a: &Tensor, opt: impl ReduceOpt) -> Tensor {
    reduce_op(a, ReduceVariance, opt)
}

impl Tensor {
    pub fn reduce_variance(&self, opt: impl ReduceOpt) -> Tensor {
        reduce_variance(self, opt)
    }
}

impl ReduceKernel<Acc> for ReduceVariance {
    #[inline]
    fn f(&self, state: Acc, x: f32) -> Acc {
        // from Welford 1962
        if state.k == 0 {
            Acc {
                k: 1,
                m: x,
                s: 0.,
            }
        } else {
            let k = state.k + 1;
            let m = state.m + (x - state.m) / k as f32;

            Acc {
                k,
                m, 
                s: state.s + (x - state.m) * (x - m),
            }
        }
    }

    #[inline]
    fn df_dx(&self, x: f32) -> f32 {
        2. * x
    }
}

struct Acc {
    k: usize,
    m: f32,
    s: f32,
}

impl reduce::State for Acc {
    type Value = f32;

    fn value(&self) -> Self::Value {
        if self.k > 1 { 
            self.s / self.k as f32
        } else {
            0.
        }
    }
}

impl Default for Acc {
    fn default() -> Self {
        Self { 
            k: 0,
            s: 0.,
            m: 0.,
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{prelude::*};

    #[test]
    fn reduce_var() {
        assert_eq!(tf32!([1.]).reduce_variance(()), tf32!(0.));
        assert_eq!(tf32!([1., 1.]).reduce_variance(()), tf32!(0.));
        assert_eq!(tf32!([2., 2., 2., 2.]).reduce_variance(()), tf32!(0.));

        assert_eq!(tf32!([1., 3.]).reduce_variance(()), tf32!(1.));
        assert_eq!(tf32!([1., 3., 1., 3.]).reduce_variance(()), tf32!(1.));
        assert_eq!(tf32!([1., 3., 3.]).reduce_variance(()), tf32!(0.8888889));
        assert_eq!(tf32!([1., 3., 4., 0.]).reduce_variance(()), tf32!(2.5));
        assert_eq!(tf32!([1., 3., 4., 0., 2.]).reduce_variance(()), tf32!(2.0));
    }
}