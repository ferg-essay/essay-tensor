use crate::{
    ops::BinaryKernel
};


#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Add;

impl BinaryKernel for Add {
    #[inline]
    fn f(&self, x: f32, y: f32) -> f32 {
        x + y
    }

    #[inline]
    fn df_dx(&self, _x: f32, _y: f32) -> f32 {
        1.
    }

    #[inline]
    fn df_dy(&self, _x: f32, _y: f32) -> f32 {
        1.
    }
}

#[cfg(test)]
mod test {
    use crate::{prelude::*, eval::{Var, Trainer}};

    #[test]
    fn test_add() {
        assert_eq!(tensor!(2.) + tensor!(3.), tensor!(5.));
        assert_eq!(tensor!([3., 4.]) + tensor!([1., 2.]), tensor!([4., 6.]));
    }

    #[test]
    fn test_add_df() {
        let x = Var::new("x", tensor!([1., 2.]));
        let y = Var::new("y", tensor!([3., 4.]));

        let module = Trainer::compile((), |()| {
            &x + &y
        }); // .training(&[&x, &y]);
        let train = module.train(());

        assert_eq!(train.value(), tensor!([4., 6.]));
        assert_eq!(train.gradient(&x), tensor!([1., 1.]));
        assert_eq!(train.gradient(&y), tensor!([1., 1.]));
    }
}
