use crate::{Tensor, ops::{reduce_op, ReduceKernel, ReduceOpt, reduce}};

#[derive(Debug, Copy, Clone)]
pub struct ReduceHypot;

pub fn reduce_hypot(a: &Tensor, opt: impl ReduceOpt) -> Tensor {
    reduce_op(a, ReduceHypot, opt)
}

impl Tensor {
    pub fn reduce_hypot(&self, opt: impl ReduceOpt) -> Tensor {
        reduce_hypot(self, opt)
    }
}

impl ReduceKernel<Acc> for ReduceHypot {
    #[inline]
    fn init(&self) -> Acc {
        Acc(0.)
    }

    #[inline]
    fn f(&self, state: Acc, a: f32) -> Acc {
        Acc(state.0 + a * a)
    }

    #[inline]
    fn df_dx(&self, a: f32) -> f32 {
        a
    }
}

struct Acc(f32);

impl reduce::State for Acc {
    type Value = f32;

    fn value(&self) -> Self::Value {
        self.0.sqrt()
    }
}

impl Default for Acc {
    fn default() -> Self {
        Acc(0.)
    }
}

#[cfg(test)]
mod test {
    use crate::{prelude::*};

    #[test]
    fn reduce_hypot() {
        assert_eq!(tf32!([1.]).reduce_mean(()), tf32!(1.));
    }
}