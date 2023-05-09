mod binary;
mod square;
mod unary;
use std::{ops};

use crate::{
    tensor::{Tensor}, 
    tensor_uop, ops::unary_op,
    tensor_binop, ops::binary_op,
};

tensor_uop!(abs, unary::Abs);
tensor_uop!(cos, unary::Cos);
tensor_uop!(exp, unary::Exp);
tensor_uop!(ln, unary::Ln);
tensor_uop!(sin, unary::Sin);

tensor_uop!(square, square::SquareOp);

tensor_binop!(atan2, binary::Atan2);
tensor_binop!(log, binary::Log);
tensor_binop!(max, binary::Max);
tensor_binop!(min, binary::Min);
tensor_binop!(powf, binary::Powf);
tensor_binop!(powi, binary::Powi);

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

tensor_ops!(Add, add, binary::Add);
tensor_ops!(Div, div, binary::Div);
tensor_ops!(Mul, mul, binary::Mul);
tensor_ops!(Rem, rem, binary::Rem);
tensor_ops!(Sub, sub, binary::Sub);

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
    unary_op(a, unary::Neg)
}

impl ops::Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        unary_op(&self, unary::Neg)
    }
}

impl ops::Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        unary_op(&self, unary::Neg)
    }
}
