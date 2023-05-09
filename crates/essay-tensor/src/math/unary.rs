use std::ops;

use crate::{Tensor, 
    module::{EvalOp}, 
    ops::{Uop, unary_op}
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Abs;

impl Uop<f32> for Abs {
    #[inline]
    fn f(&self, value: f32) -> f32 {
        value.abs()
    }

    #[inline]
    fn df_dx(&self, value: f32) -> f32 {
        if value > 0. { 1. } else { -1. }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cos;

impl Uop<f32> for Cos {
    #[inline]
    fn f(&self, value: f32) -> f32 {
        value.cos()
    }

    #[inline]
    fn df_dx(&self, value: f32) -> f32 {
        value.sin()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Exp;

impl Uop<f32> for Exp {
    #[inline]
    fn f(&self, value: f32) -> f32 {
        value.exp()
    }

    #[inline]
    fn df_dx(&self, value: f32) -> f32 {
        value.exp()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ln;

impl Uop<f32> for Ln {
    #[inline]
    fn f(&self, value: f32) -> f32 {
        value.ln()
    }

    #[inline]
    fn df_dx(&self, value: f32) -> f32 {
        1. / value
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sin;

impl Uop<f32> for Sin {
    #[inline]
    fn f(&self, value: f32) -> f32 {
        value.sin()
    }

    #[inline]
    fn df_dx(&self, value: f32) -> f32 {
        - value.cos()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Unary {
    Abs,
    Cos,
    Exp,
    Ln,
    Mul(f32),
    Neg,
    Sin,
}

impl Uop<f32> for Unary {
    fn f(&self, value: f32) -> f32 {
        match &self {
            Unary::Abs => value.abs(),
            Unary::Cos => value.cos(),
            Unary::Exp => value.exp(),
            Unary::Ln => value.ln(),
            Unary::Mul(a) => a * value,
            Unary::Neg => -value,
            Unary::Sin => value.sin(),
        }
    }

    fn df_dx(&self, _value: f32) -> f32 {
        todo!()
    }
}

pub fn neg(a: &Tensor) -> Tensor {
    unary_op(a, Unary::Neg)
}

impl ops::Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        unary_op(&self, Unary::Neg)
    }
}

impl ops::Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        unary_op(&self, Unary::Neg)
    }
}

impl EvalOp for Unary {
    fn eval(
        &self,
        args: &[&Tensor],
    ) -> Tensor {
        match self {
            Unary::Abs => args[0].abs(),
            Unary::Cos => args[0].cos(),
            Unary::Exp => args[0].exp(),
            Unary::Ln => args[0].ln(),
            Unary::Mul(a) => (1. / *a) * args[0],
            Unary::Neg => neg(args[0]),
            Unary::Sin => args[0].sin(),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::{*};

    #[test]
    fn neg_f() {
        assert_eq!(- tensor!(1.), tensor!(-1.));
        assert_eq!(- tensor!([1.]), tensor!([-1.]));
        assert_eq!(- tensor!([[1.]]), tensor!([[-1.]]));
        assert_eq!(- tensor!([[[1.]]]), tensor!([[[-1.]]]));
        assert_eq!(- tensor!([[[[1.]]]]), tensor!([[[[-1.]]]]));

        assert_eq!(- tensor!([1., 2.]), tensor!([-1., -2.]));
        assert_eq!(- tensor!([[-1., -2.], [-3., -4.]]), 
            tensor!([[1., 2.], [3., 4.]]));
    }
}