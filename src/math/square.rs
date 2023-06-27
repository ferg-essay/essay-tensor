use crate::{ops::UnaryKernel};

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct SquareOp;

impl UnaryKernel<f32> for SquareOp {
    #[inline]
    fn f(&self, value: f32) -> f32 {
        value.powi(2)
    }

    #[inline]
    fn df_dx(&self, value: f32) -> f32 {
        2. * value
    }
}

#[cfg(test)]
mod test {
    use crate::{prelude::*, model::{Var, Trainer}};

    #[test]
    fn square() {
        assert_eq!(tensor!(2.).square(), tensor!(4.));
        assert_eq!(tensor!([3., 4.]).square(), tensor!([9., 16.]));
    }

    #[test]
    fn square_df() {
        let x = Var::new("x", tensor!([1., 2., 3.]));

        let module: Trainer<(), Tensor> = Trainer::compile((), |(), _| {
            x.square()
        }); // .training(&[&x]);
        let train = module.train(());

        assert_eq!(train.value(), tensor!([1., 4., 9.]));
        assert_eq!(train.gradient(&x), tensor!([2., 4., 6.]));
    }
}
