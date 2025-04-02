mod sin;
mod neg;
mod clamp;
mod ln;
mod exp;
mod cos;
mod abs;
mod hypot;
mod log;
mod min;
mod max;
mod normalize_unit;
mod powi;
mod powf;
mod reduce_hypot;
mod reduce_mean;
mod reduce_min;
mod reduce_max;
mod reduce_sum;
mod reduce_std;
mod reduce_variance;
mod rem;
mod sub;
mod mul;
mod div;
mod atan2;
pub(crate) mod add;
mod square;
use std::ops;

use crate::{
    tensor::Tensor, 
    tensor_uop, ops::unary_op,
    tensor_binop, ops::binary_op,
};

// tensor_uop!(abs, abs::Abs);
tensor_uop!(cos, cos::Cos);
//tensor_uop!(exp, exp::Exp);
//tensor_uop!(ln, ln::Ln);
tensor_uop!(sin, sin::Sin);

tensor_uop!(square, square::SquareOp);

tensor_binop!(atan2, atan2::Atan2);
tensor_binop!(hypot, hypot::Hypot);
//tensor_binop!(log, log::Log);
//tensor_binop!(max, max::Max);
//tensor_binop!(min, min::Min);
//tensor_binop!(powf, powf::Powf);
//tensor_binop!(powi, powi::Powi);

pub use reduce_hypot::reduce_hypot;
pub use reduce_mean::reduce_mean;
pub use reduce_min::reduce_min;
pub use reduce_max::reduce_max;
pub use reduce_sum::{reduce_sum, reduce_sum_opt};
pub use reduce_std::reduce_std;
pub use reduce_variance::reduce_variance;

pub use normalize_unit::normalize_unit;

//
// overloaded operations: Add, Sub, Mul
//

macro_rules! tensor_ops {
    ($ty:ty, $op:ident, $fun:ident, $binop:expr, $uop_st:ty, $uop_ts:ty) => {
        pub fn $fun(a: impl Into<Tensor<$ty>>, b: impl Into<Tensor<$ty>>) -> Tensor<$ty> {
            let a = a.into();
            let b = b.into();

            binary_op(&a, &b, $binop)
        }

        impl ops::$op<Tensor<$ty>> for Tensor<$ty> {
            type Output = Tensor<$ty>;
        
            fn $fun(self, rhs: Tensor<$ty>) -> Self::Output {
                binary_op(&self, &rhs, $binop)
            }
        }

        impl ops::$op<Tensor<$ty>> for &Tensor<$ty> {
            type Output = Tensor<$ty>;
        
            fn $fun(self, rhs: Tensor<$ty>) -> Self::Output {
                binary_op(self, &rhs, $binop)
            }
        }

        impl ops::$op<Tensor<$ty>> for $ty {
            type Output = Tensor<$ty>;
        
            fn $fun(self, rhs: Tensor<$ty>) -> Self::Output {
                //binary_op(&Tensor::<$ty>::from(self), &rhs, $binop)
                unary_op(&rhs, <$uop_st>::new(self))
            }
        }

        impl ops::$op<&Tensor<$ty>> for Tensor<$ty> {
            type Output = Tensor<$ty>;
        
            fn $fun(self, rhs: &Tensor<$ty>) -> Self::Output {
                binary_op(&self, rhs, $binop)
            }
        }

        impl ops::$op<&Tensor<$ty>> for &Tensor<$ty> {
            type Output = Tensor<$ty>;
        
            fn $fun(self, rhs: &Tensor<$ty>) -> Self::Output {
                binary_op(self, rhs, $binop)
            }
        }

        impl ops::$op<&Tensor<$ty>> for $ty {
            type Output = Tensor<$ty>;
        
            fn $fun(self, rhs: &Tensor<$ty>) -> Self::Output {
                // binary_op(&Tensor::<$ty>::from(self), &rhs, $binop)
                unary_op(&rhs, <$uop_st>::new(self))
            }
        }

        impl ops::$op<$ty> for Tensor<$ty> {
            type Output = Tensor<$ty>;
        
            fn $fun(self, rhs: $ty) -> Self::Output {
                // binary_op(&self, &Tensor::from(rhs), $binop)
                unary_op(&self, <$uop_ts>::new(rhs))
            }
        }

        impl ops::$op<$ty> for &Tensor<$ty> {
            type Output = Tensor<$ty>;
        
            fn $fun(self, rhs: $ty) -> Self::Output {
                // binary_op(&self, &Tensor::from(rhs), $binop)
                unary_op(&self, <$uop_ts>::new(rhs))
            }
        }
    }
}

macro_rules! _tensor_ops2 {
    ($ty:ty, $op:ident, $fun:ident, $binop:expr, $uop_st:ty, $uop_ts:ty) => {
        pub fn $fun<D>(
            a: impl Into<Tensor<D>>, 
            b: impl Into<Tensor<D>>
        ) -> Tensor<D> 
        where
            D: Dtype + ops::$op<Output=D> + Copy
        {
            let a = a.into();
            let b = b.into();

            binary_op(&a, &b, $binop)
        }

        impl<D> ops::$op<Tensor<D>> for Tensor<D>
        where
            D: Dtype + ops::$op<Output=D> + Copy
        {
            type Output = Tensor<D>;
        
            fn $fun(self, rhs: Tensor<D>) -> Self::Output {
                binary_op(&self, &rhs, $binop)
            }
        }

        impl<D> ops::$op<Tensor<D>> for &Tensor<D> 
        where
            D: Dtype + ops::$op<Output=D> + Copy
        {
            type Output = Tensor<D>;
        
            fn $fun(self, rhs: Tensor<D>) -> Self::Output {
                binary_op(self, &rhs, $binop)
            }
        }

        impl<D> ops::$op<&Tensor<D>> for Tensor<D>
        where
            D: Dtype + ops::$op<Output=D> + Copy
        {
            type Output = Tensor<D>;
        
            fn $fun(self, rhs: &Tensor<D>) -> Self::Output {
                binary_op(&self, rhs, $binop)
            }
        }

        impl<D> ops::$op<&Tensor<D>> for &Tensor<D> 
        where
            D: Dtype + ops::$op<Output=D> + Copy
        {
            type Output = Tensor<D>;
        
            fn $fun(self, rhs: &Tensor<D>) -> Self::Output {
                binary_op(self, rhs, $binop)
            }
        }

        impl ops::$op<Tensor<$ty>> for $ty
        {
            type Output = Tensor<$ty>;
        
            fn $fun(self, rhs: Tensor<$ty>) -> Self::Output {
                //binary_op(&Tensor::<$ty>::from(self), &rhs, $binop)
                unary_op(&rhs, <$uop_st>::new(self))
            }
        }

        impl ops::$op<&Tensor<$ty>> for $ty {
            type Output = Tensor<$ty>;
        
            fn $fun(self, rhs: &Tensor<$ty>) -> Self::Output {
                // binary_op(&Tensor::<$ty>::from(self), &rhs, $binop)
                unary_op(&rhs, <$uop_st>::new(self))
            }
        }

        impl ops::$op<$ty> for Tensor<$ty> {
            type Output = Tensor<$ty>;
        
            fn $fun(self, rhs: $ty) -> Self::Output {
                // binary_op(&self, &Tensor::from(rhs), $binop)
                unary_op(&self, <$uop_ts>::new(rhs))
            }
        }

        impl ops::$op<$ty> for &Tensor<$ty> {
            type Output = Tensor<$ty>;
        
            fn $fun(self, rhs: $ty) -> Self::Output {
                // binary_op(&self, &Tensor::from(rhs), $binop)
                unary_op(&self, <$uop_ts>::new(rhs))
            }
        }
    }
}

//tensor_ops2!(f32, Add, add, add::Add2::new(), add::AddScalar, add::AddScalar);
tensor_ops!(f32, Div, div, div::Div, div::DivST, div::DivTS);
tensor_ops!(f32, Mul, mul, mul::Mul, mul::MulScalar, mul::MulScalar);
tensor_ops!(f32, Rem, rem, rem::Rem, rem::RemST, rem::RemTS);
tensor_ops!(f32, Sub, sub, sub::Sub, sub::SubST, sub::SubTS);

//tensor_ops!(C32, Add, add, add::Add::new(), add::AddScalar, add::AddScalar);

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

// clamp

pub fn clamp(x: &Tensor, min: f32, max: f32) -> Tensor {
    unary_op(x, clamp::ClampScalar::new(min, max))
}

impl Tensor {
    pub fn clamp(&self, min: f32, max: f32) -> Tensor {
        clamp(self, min, max)
    }
}
