mod binary;
mod square;
mod unary;
use std::{ops, any::type_name};

use crate::{
    tensor::{Tensor}, 
    tensor_uop, ops::unary_op,
    tensor_binop, ops::binary_op,
    math::binary::Binary,
};

tensor_uop!(abs, unary::Abs);
tensor_uop!(cos, unary::Cos);
tensor_uop!(exp, unary::Exp);
tensor_uop!(ln, unary::Ln);
tensor_uop!(sin, unary::Sin);

tensor_uop!(square, square::SquareOp);

tensor_binop!(atan2, Binary::Atan2);
tensor_binop!(log, Binary::Log);
tensor_binop!(max, Binary::Max);
tensor_binop!(min, Binary::Min);
tensor_binop!(powf, Binary::Powf);
tensor_binop!(powi, Binary::Powi);

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
tensor_ops!(Rem, rem, Binary::Rem);
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
