mod reduce_mean;
mod sin;
mod neg;
mod ln;
mod exp;
mod cos;
mod abs;
mod powi;
mod powf;
mod min;
mod max;
mod log;
mod sub;
mod reduce_sum;
mod rem;
mod mul;
mod div;
mod atan2;
pub(crate) mod add;
mod square;
use std::{ops};

use crate::{
    tensor::{Tensor}, 
    tensor_uop, ops::unary_op,
    tensor_binop, ops::binary_op,
};

tensor_uop!(abs, abs::Abs);
tensor_uop!(cos, cos::Cos);
tensor_uop!(exp, exp::Exp);
tensor_uop!(ln, ln::Ln);
tensor_uop!(sin, sin::Sin);

tensor_uop!(square, square::SquareOp);

tensor_binop!(atan2, atan2::Atan2);
tensor_binop!(log, log::Log);
tensor_binop!(max, max::Max);
tensor_binop!(min, min::Min);
tensor_binop!(powf, powf::Powf);
tensor_binop!(powi, powi::Powi);

pub use reduce_sum::{reduce_sum, reduce_sum_opt};

//
// overloaded operations: Add, Sub, Mul
//

macro_rules! tensor_ops {
    ($op:ident, $fun:ident, $binop:expr) => {
        pub fn $fun(a: impl Into<Tensor>, b: impl Into<Tensor>) -> Tensor {
            let a = a.into();
            let b = b.into();

            binary_op(&a, &b, $binop)
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

tensor_ops!(Add, add, add::Add);
tensor_ops!(Div, div, div::Div);
tensor_ops!(Mul, mul, mul::Mul);
tensor_ops!(Rem, rem, rem::Rem);
tensor_ops!(Sub, sub, sub::Sub);

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

//
// neg
//

pub fn neg(a: &Tensor) -> Tensor {
    unary_op(a, neg::Neg)
}

impl ops::Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        unary_op(&self, neg::Neg)
    }
}

impl ops::Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        unary_op(&self, neg::Neg)
    }
}
