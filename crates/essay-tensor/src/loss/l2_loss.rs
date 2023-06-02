use reduce::{ReduceOpt, ReduceKernel};

use crate::{Tensor, ops::{reduce_op, reduce}};

#[derive(Debug, Copy, Clone)]
pub struct L2Loss;

pub fn l2_loss(a: impl Into<Tensor>) -> Tensor {
    l2_loss_opt(a, ())
}

pub fn l2_loss_opt(a: impl Into<Tensor>, opt: impl ReduceOpt) -> Tensor {
    reduce_op(a, L2Loss, opt)
}

impl Tensor {
    pub fn l2_loss(&self) -> Tensor {
        l2_loss_opt(self, ())
    }
}

impl ReduceKernel<f32> for L2Loss {
    #[inline]
    fn f(&self, acc: f32, x: f32) -> f32 {
        acc + 0.5 * x * x
    }

    fn df_dx(&self, x: f32) -> f32 {
        x
    }
}

#[cfg(test)]
mod test {
    use crate::{prelude::*, model::Var};

    #[test]
    fn l2_loss() {
        assert_eq!(tensor!(0.).l2_loss(), tensor!(0.));
        assert_eq!(tensor!(2.).l2_loss(), tensor!(2.));
        assert_eq!(tensor!(3.).l2_loss(), tensor!(4.5));
    }

    #[test]
    fn l2_loss_rank2() {
        assert_eq!(tf32!([[1.]]).l2_loss(), tf32!(0.5));
        assert_eq!(tf32!([[0.], [1.]]).l2_loss(), tf32!(0.5));
        assert_eq!(tf32!([[0., 1.]]).l2_loss(), tf32!(0.5));
        assert_eq!(tf32!([[1., 0.], [0., 1.]]).l2_loss(), tf32!(1.0));
    }

    #[test]
    fn l2_loss_df_n() {
        let x = Var::new("x", tf32!([1., 2., 2., 1.]));

        let module = Trainer::compile((), |(), _| {
            2. * x.l2_loss()
        });
        let train = module.train(());

        assert_eq!(train.value(), tf32!(10.));
        assert_eq!(train.gradient(&x), tf32!([2., 4., 4., 2.]));
    }
}
