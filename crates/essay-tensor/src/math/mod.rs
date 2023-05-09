mod square;
mod unary;
use std::{ops, any::type_name};

use crate::{
    tensor::{Tensor}, 
    module::{TensorId, Graph, TensorCache, graph::BackOp}, 
    tensor_uop, ops::unary_op,
    tensor_binop, ops::binary_op,
    math::unary::Unary, ops::Binop
};

tensor_uop!(abs, Unary::Abs);
tensor_uop!(cos, Unary::Cos);
tensor_uop!(exp, Unary::Exp);
tensor_uop!(ln, Unary::Ln);
tensor_uop!(sin, Unary::Sin);

tensor_uop!(square, square::SquareOp);

//
// binops
//

#[derive(Debug, Clone, PartialEq)]
pub struct Add;

impl Binop for Add {
    fn f(&self, x: f32, y: f32) -> f32 {
        x + y
    }

    fn df_dx(&self, _x: f32, _y: f32) -> f32 {
        1.
    }

    fn df_dy(&self, _x: f32, _y: f32) -> f32 {
        1.
    }
}

#[derive(Debug, Clone)]
enum Binary {
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
/*
impl BackOp for Binary {
    fn name(&self) -> &str {
        type_name::<Self>()
    }

    fn df(
        &self,
        _tensors: &TensorCache,
        args: &[&Tensor],
        prev: &Tensor,
    ) -> Tensor {
        match self {
            Binary::Mul => {
                args[0].eval_binop(prev, self)
            },
            _ => todo!("unimplemented back {:?}", self),
        }
    }
}
*/

tensor_binop!(atan2, Binary::Atan2);
tensor_binop!(div_euclid, Binary::DivEuclid);
tensor_binop!(hypot, Binary::Hypot);
tensor_binop!(log, Binary::Log);
tensor_binop!(max, Binary::Max);
tensor_binop!(min, Binary::Min);
tensor_binop!(powf, Binary::Powf);
tensor_binop!(powi, Binary::Powi);
tensor_binop!(rem_euclid, Binary::RemEuclid);

//
// overloaded operations: Add, Sub, Mul
//

macro_rules! tensor_ops {
    ($op:ident, $fun:ident, $binop:expr) => {
        pub fn $fun(a: &Tensor, b: &Tensor) -> Tensor {
            binary_op(a, b, $binop)
        }

        impl ops::$op for &Tensor {
            type Output = Tensor;
        
            fn $fun(self, rhs: Self) -> Self::Output {
                binary_op(self, &rhs, $binop)
            }
        }

        impl ops::$op for Tensor {
            type Output = Tensor;
        
            fn $fun(self, rhs: Self) -> Self::Output {
                binary_op(&self, &rhs, $binop)
            }
        }

        impl ops::$op<&Tensor> for Tensor {
            type Output = Tensor;
        
            fn $fun(self, rhs: &Tensor) -> Self::Output {
                binary_op(&self, &rhs, $binop)
            }
        }

        impl ops::$op<Tensor> for &Tensor {
            type Output = Tensor;
        
            fn $fun(self, rhs: Tensor) -> Self::Output {
                binary_op(self, &rhs, $binop)
            }
        }

        impl ops::$op<&Tensor> for f32 {
            type Output = Tensor;
        
            fn $fun(self, rhs: &Tensor) -> Self::Output {
                binary_op(&Tensor::from(self), &rhs, $binop)
            }
        }

        impl ops::$op<Tensor> for f32 {
            type Output = Tensor;
        
            fn $fun(self, rhs: Tensor) -> Self::Output {
                binary_op(&Tensor::from(self), &rhs, $binop)
            }
        }
    }
}

tensor_ops!(Add, add, Add);
tensor_ops!(Div, div, Binary::Div);
tensor_ops!(Mul, mul, Binary::Mul);
tensor_ops!(Rem, rem, Binary::Rem);
tensor_ops!(Sub, sub, Binary::Sub);

impl ops::Mul<Option<Tensor>> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Option<Tensor>) -> Self::Output {
        match rhs {
            Some(rhs) => self * rhs,
            None => self,
        }
    }
}

impl ops::Add<Option<Tensor>> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: Option<Tensor>) -> Self::Output {
        match rhs {
            Some(rhs) => self + rhs,
            None => self.clone(),
        }
    }
}

impl ops::Add<Option<Tensor>> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Option<Tensor>) -> Self::Output {
        match rhs {
            Some(rhs) => self + rhs,
            None => self,
        }
    }
}

impl ops::Mul<Option<Tensor>> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Option<Tensor>) -> Self::Output {
        match rhs {
            Some(rhs) => self * rhs,
            None => self.clone(),
        }
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
