
//
// binops
//

use crate::ops::Binop;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Add;

impl Binop for Add {
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Atan2;

impl Binop for Atan2 {
    #[inline]
    fn f(&self, x: f32, y: f32) -> f32 {
        x.atan2(y)
    }

    #[inline]
    fn df_dx(&self, _x: f32, _y: f32) -> f32 {
        todo!()
    }

    #[inline]
    fn df_dy(&self, _x: f32, _y: f32) -> f32 {
        todo!()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Div;

impl Binop for Div {
    #[inline]
    fn f(&self, x: f32, y: f32) -> f32 {
        x / y
    }

    #[inline]
    fn df_dx(&self, _x: f32, y: f32) -> f32 {
        1. / y
    }

    #[inline]
    fn df_dy(&self, x: f32, y: f32) -> f32 {
        - x / (y * y)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rem;

impl Binop for Rem {
    #[inline]
    fn f(&self, x: f32, y: f32) -> f32 {
        x % y
    }

    #[inline]
    fn df_dx(&self, _x: f32, _y: f32) -> f32 {
        unimplemented!();
    }

    #[inline]
    fn df_dy(&self, _x: f32, _y: f32) -> f32 {
        unimplemented!();
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sub;

impl Binop for Sub {
    #[inline]
    fn f(&self, x: f32, y: f32) -> f32 {
        x - y
    }

    #[inline]
    fn df_dx(&self, _x: f32, _y: f32) -> f32 {
        1.
    }

    #[inline]
    fn df_dy(&self, _x: f32, _y: f32) -> f32 {
        -1.
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Log;

impl Binop for Log {
    #[inline]
    fn f(&self, x: f32, y: f32) -> f32 {
        x.log(y)
    }

    #[inline]
    fn df_dx(&self, _x: f32, y: f32) -> f32 {
        todo!()
    }

    #[inline]
    fn df_dy(&self, x: f32, _y: f32) -> f32 {
        todo!()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Max;

impl Binop for Max {
    #[inline]
    fn f(&self, x: f32, y: f32) -> f32 {
        x.max(y)
    }

    #[inline]
    fn df_dx(&self, _x: f32, y: f32) -> f32 {
        todo!()
    }

    #[inline]
    fn df_dy(&self, x: f32, _y: f32) -> f32 {
        todo!()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Min;

impl Binop for Min {
    #[inline]
    fn f(&self, x: f32, y: f32) -> f32 {
        x.min(y)
    }

    #[inline]
    fn df_dx(&self, _x: f32, y: f32) -> f32 {
        todo!()
    }

    #[inline]
    fn df_dy(&self, x: f32, _y: f32) -> f32 {
        todo!()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mul;

impl Binop for Mul {
    #[inline]
    fn f(&self, x: f32, y: f32) -> f32 {
        x * y
    }

    #[inline]
    fn df_dx(&self, _x: f32, y: f32) -> f32 {
        y
    }

    #[inline]
    fn df_dy(&self, x: f32, _y: f32) -> f32 {
        x
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Powf;

impl Binop for Powf {
    #[inline]
    fn f(&self, x: f32, y: f32) -> f32 {
        x.powf(y)
    }

    #[inline]
    fn df_dx(&self, _x: f32, y: f32) -> f32 {
        todo!()
    }

    #[inline]
    fn df_dy(&self, x: f32, _y: f32) -> f32 {
        todo!()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Powi;

impl Binop for Powi {
    #[inline]
    fn f(&self, x: f32, y: f32) -> f32 {
        x.powi(y as i32)
    }

    #[inline]
    fn df_dx(&self, _x: f32, y: f32) -> f32 {
        todo!()
    }

    #[inline]
    fn df_dy(&self, x: f32, _y: f32) -> f32 {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use crate::{prelude::*, module::{Var, Module}};

    #[test]
    fn test_add() {
        assert_eq!(tensor!(2.) + tensor!(3.), tensor!(5.));
        assert_eq!(tensor!([3., 4.]) + tensor!([1., 2.]), tensor!([4., 6.]));
    }

    #[test]
    fn test_add_df() {
        let x = Var::new("x", tensor!([1., 2.]));
        let y = Var::new("y", tensor!([3., 4.]));

        let module = Module::build((), |()| {
            &x + &y
        }).training(&[&x, &y]);
        let train = module.train(());

        assert_eq!(train.value(), tensor!([4., 6.]));
        assert_eq!(train.gradient(&x), tensor!([1., 1.]));
        assert_eq!(train.gradient(&y), tensor!([1., 1.]));
    }
}
