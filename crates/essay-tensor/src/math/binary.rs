
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

#[derive(Debug, Clone)]
pub enum Binary {
    Add,
    Atan2,
    Div,
    DivEuclid,
    Hypot,
    Log,
    Max,
    Min,
    Mul,
    Powf,
    Powi,
    Rem,
    RemEuclid,
    Sub,
}

impl Binop<f32> for Binary {
    #[inline]
    fn f(&self, a: f32, b: f32) -> f32 {
        match &self {
            Binary::Add => a + b,
            Binary::Atan2 => a.atan2(b),
            Binary::Div => a / b,
            Binary::DivEuclid => a.div_euclid(b),
            Binary::Hypot => a.hypot(b),
            Binary::Log => a.log(b),
            Binary::Max => a.max(b),
            Binary::Min => a.min(b),
            Binary::Mul => a * b,
            Binary::Powf => a.powf(b),
            Binary::Powi => a.powi(b as i32),
            Binary::Rem => a % b,
            Binary::RemEuclid => a.rem_euclid(b),
            Binary::Sub => a - b,
        }
    }

    fn df_dx(&self, x: f32, y: f32) -> f32 {
        todo!()
    }

    fn df_dy(&self, x: f32, y: f32) -> f32 {
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
