mod reduce;
mod reduce_std;
use std::ops;

use num_traits::{Float, Num, One, Signed, Zero};

use crate::tensor::{Shape, Tensor, Type};

/*
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
*/
// pub use reduce::reduce_axis;
pub use reduce_std::reduce_std;
/*
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
*/

impl<T: Type + Zero + Clone> Tensor<T> {
    #[inline]
    pub fn zeros(shape: impl Into<Shape>) -> Self {
        Self::init(shape, || Zero::zero())
    }
}

impl<T: Type + One + Clone> Tensor<T> {
    #[inline]
    pub fn ones(shape: impl Into<Shape>) -> Self {
        Self::init(shape, || One::one())
    }
}

impl<T: Type + Signed + Clone> Tensor<T> {
    #[inline]
    pub fn abs(&self) -> Self {
        self.map(|v| v.abs())
    }

    #[inline]    
    pub fn abs_sub(&self, rhs: &Self) -> Self {
        self.map2(rhs, |a, b| a.abs_sub(b))
    }

    #[inline]
    pub fn neg(&self) -> Self {
        self.map(|v| v.clone().neg())
    }

    #[inline]
    pub fn is_negative(&self) -> Tensor<bool> {
        self.map(|v| v.is_negative())
    }

    #[inline]
    pub fn is_positive(&self) -> Tensor<bool> {
        self.map(|v| v.is_positive())
    }
}

impl<T: Type + Ord + Clone> Tensor<T> {
    #[inline]
    pub fn min(&self, rhs: impl Into<Tensor<T>>) -> Self {
        self.map2(&rhs.into(), |a, b| a.clone().min(b.clone()))
    }

    #[inline]
    pub fn max(&self, rhs: impl Into<Tensor<T>>) -> Self {
        self.map2(&rhs.into(), |a, b| a.clone().max(b.clone()))
    }
}

impl<T: Type + Float + Clone> Tensor<T> {
    #[inline]
    pub fn floor(&self) -> Tensor<T> {
        self.map(|v| v.floor())
    }

    #[inline]
    pub fn ceil(&self) -> Tensor<T> {
        self.map(|v| v.ceil())
    }

    #[inline]
    pub fn round(&self) -> Tensor<T> {
        self.map(|v| v.round())
    }

    #[inline]
    pub fn trunc(&self) -> Tensor<T> {
        self.map(|v| v.trunc())
    }

    #[inline]
    pub fn fract(&self) -> Tensor<T> {
        self.map(|v| v.fract())
    }

    #[inline]
    pub fn signum(&self) -> Tensor<T> {
        self.map(|v| v.signum())
    }

    #[inline]
    pub fn recip(&self) -> Tensor<T> {
        self.map(|v| v.recip())
    }

    #[inline]
    pub fn powi(&self, rhs: impl Into<Tensor<i32>>) -> Tensor<T> {
        self.map2(&rhs.into(), |a, b| a.powi(*b))
    }

    #[inline]
    pub fn powf(&self, rhs: impl Into<Tensor<T>>) -> Tensor<T> {
        self.map2(&rhs.into(), |a, b| a.powf(*b))
    }

    #[inline]
    pub fn sqrt(&self) -> Tensor<T> {
        self.map(|a| a.sqrt())
    }

    #[inline]
    pub fn exp(&self) -> Tensor<T> {
        self.map(|a| a.exp())
    }

    #[inline]
    pub fn exp2(&self) -> Tensor<T> {
        self.map(|a| a.exp2())
    }

    #[inline]
    pub fn ln(&self) -> Tensor<T> {
        self.map(|a| a.ln())
    }

    #[inline]
    pub fn log(&self, base: impl Into<Tensor<T>>) -> Tensor<T> {
        self.map2(&base.into(), |a, b| a.log(*b))
    }

    #[inline]
    pub fn log2(&self) -> Tensor<T> {
        self.map(|a| a.log2())
    }

    #[inline]
    pub fn log10(&self) -> Tensor<T> {
        self.map(|a| a.log10())
    }

    #[inline]
    pub fn to_degrees(&self) -> Tensor<T> {
        self.map(|a| a.to_degrees())
    }

    #[inline]
    pub fn to_radians(&self) -> Tensor<T> {
        self.map(|a| a.to_radians())
    }
}

macro_rules! tensor_ops2 {
    ($ty:ident, $op:ident, $fun:ident, $binop:ident, $uop_st:ident, $uop_ts:ident) => {
        impl<T: $ty + Type + Clone> ops::$op<Tensor<T>> for Tensor<T>
        {
            type Output = Tensor<T>;
        
            fn $fun(self, rhs: Tensor<T>) -> Self::Output {
                $binop(&self, &rhs)
            }
        }

        impl<T: $ty + Type + Clone> ops::$op<Tensor<T>> for &Tensor<T> 
        {
            type Output = Tensor<T>;
        
            fn $fun(self, rhs: Tensor<T>) -> Self::Output {
                $binop(self, &rhs)
            }
        }

        impl<T: $ty + Type + Clone> ops::$op<&Tensor<T>> for Tensor<T>
        {
            type Output = Tensor<T>;
        
            fn $fun(self, rhs: &Tensor<T>) -> Self::Output {
                $binop(&self, rhs)
            }
        }

        impl<T: $ty + Type + Clone> ops::$op<&Tensor<T>> for &Tensor<T> 
        {
            type Output = Tensor<T>;
        
            fn $fun(self, rhs: &Tensor<T>) -> Self::Output {
                $binop(self, rhs)
            }
        }

    impl<T: $ty + Type + Clone> ops::$op<T> for Tensor<T> {
        type Output = Tensor<T>;
    
        fn $fun(self, rhs: T) -> Self::Output {
            $uop_ts(&self, rhs)
        }
    }

    impl<T: $ty + Type + Clone> ops::$op<T> for &Tensor<T> {
        type Output = Tensor<T>;
    
        fn $fun(self, rhs: T) -> Self::Output {
            $uop_ts(&self, rhs)
        }
    }
}}

macro_rules! tensor_ops2_scalar {
    ($ty:ty, $op:ident, $fun:ident, $binop:ident, $uop_st:ident, $uop_ts:ident) => {
        impl ops::$op<Tensor<$ty>> for $ty
        {
            type Output = Tensor<$ty>;
        
            fn $fun(self, rhs: Tensor<$ty>) -> Self::Output {
                $uop_st(self, &rhs)
            }
        }

        impl ops::$op<&Tensor<$ty>> for $ty {
            type Output = Tensor<$ty>;
        
            fn $fun(self, rhs: &Tensor<$ty>) -> Self::Output {
                $uop_st(self, rhs)
            }
        }
    }
}

fn add_f32<T: Type + Num + Clone>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
    a.map2(b, |a, b| a.clone() + b.clone())
}

fn add_f32_ts<T: Type + Num + Clone>(a: &Tensor<T>, b: T) -> Tensor<T> {
    a.map(|a| a.clone() + b.clone())
}

fn add_f32_st<T: Type + Num + Clone>(a: T, b: &Tensor<T>) -> Tensor<T> {
    b.map(|b| a.clone() + b.clone())
}

tensor_ops2!(Num, Add, add, add_f32, add_f32_st, add_f32_ts);
tensor_ops2_scalar!(f32, Add, add, add_f32, add_f32_st, add_f32_ts);
tensor_ops2_scalar!(i32, Add, add, add_32, add_f32_st, add_f32_ts);

fn mul<T: Type + Num + Clone>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
    a.map2(b, |a, b| a.clone() * b.clone())
}

fn mul_ts<T: Type + Num + Clone>(a: &Tensor<T>, b: T) -> Tensor<T> {
    a.map(|a| a.clone() * b.clone())
}

fn mul_st<T: Type + Num + Clone>(a: T, b: &Tensor<T>) -> Tensor<T> {
    b.map(|b| a.clone() * b.clone())
}

tensor_ops2!(Num, Mul, mul, mul, mul_st, mul_ts);
tensor_ops2_scalar!(f32, Mul, mul, mul, mul_st, mul_ts);
tensor_ops2_scalar!(i32, Mul, mul, mul, mul_st, mul_ts);

fn div<T: Type + Num + Clone>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
    a.map2(b, |a, b| a.clone() / b.clone())
}

fn div_ts<T: Type + Num + Clone>(a: &Tensor<T>, b: T) -> Tensor<T> {
    a.map(|a| a.clone() / b.clone())
}

fn div_st<T: Type + Num + Clone>(a: T, b: &Tensor<T>) -> Tensor<T> {
    b.map(|b| a.clone() / b.clone())
}

tensor_ops2!(Num, Div, div, div, div_st, div_ts);
tensor_ops2_scalar!(f32, Div, div, div, div_st, div_ts);
tensor_ops2_scalar!(i32, Div, div, div, div_st, div_ts);

fn rem<T: Type + Num + Clone>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
    a.map2(b, |a, b| a.clone() % b.clone())
}

fn rem_ts<T: Type + Num + Clone>(a: &Tensor<T>, b: T) -> Tensor<T> {
    a.map(|a| a.clone() % b.clone())
}

fn rem_st<T: Type + Num + Clone>(a: T, b: &Tensor<T>) -> Tensor<T> {
    b.map(|b| a.clone() % b.clone())
}

tensor_ops2!(Num, Rem, rem, rem, rem_st, rem_ts);
tensor_ops2_scalar!(f32, Rem, rem, rem, rem_st, rem_ts);
tensor_ops2_scalar!(i32, Rem, rem, rem, rem_st, rem_ts);


#[cfg(test)]
mod test {
    use crate::{ten, tf32};

    #[test]
    fn abs() {
        assert_eq!(ten!(-1.).abs(), ten!(1.));
        assert_eq!(ten!([1., -2., 3.]).abs(), ten!([1., 2., 3.]));
        assert_eq!(
            ten!([[1., -2.], [-3., 4.]]).abs(), 
            ten!([[1., 2.], [3., 4.]]),
        );
        assert_ne!(
            ten!([[-1., 2.], [3., -4.]]).abs(), 
            ten!([1., 2., 3., 4.]),
        );
        assert_eq!(
            ten!([[1., 2.], [-3., 4.]]).subslice(1, 1).abs(), 
            ten!([[3., 4.]]),
        );
    }

    #[test]
    fn abs_i32() {
        assert_eq!(ten!(-1i32).abs(), ten!(1i32));
        assert_eq!(ten!([1, -2, 3]).abs(), ten!([1, 2, 3]));
        assert_eq!(
            ten!([[1, -2], [-3, 4]]).abs(), 
            ten!([[1, 2], [3, 4]]),
        );
        assert_ne!(
            ten!([[-1, 2], [3, -4]]).abs(), 
            ten!([1, 2, 3, 4]),
        );
        assert_eq!(
            ten!([[1, 2], [-3, 4]]).subslice(1, 1).abs(), 
            ten!([[3, 4]]),
        );
    }

    #[test]
    fn neg() {
        assert_eq!(ten!(1.).neg(), ten!(-1.));
        assert_eq!(ten!([1., 2., 3.]).neg(), ten!([-1., -2., -3.]));
        assert_eq!(
            ten!([[1., 2.], [3., 4.]]).neg(), 
            ten!([[-1., -2.], [-3., -4.]]),
        );
        assert_ne!(
            ten!([[1., 2.], [3., 4.]]).neg(), 
            ten!([-1., -2., -3., -4.]),
        );
        assert_eq!(
            ten!([[1., 2.], [3., 4.]]).subslice(1, 1).neg(), 
            ten!([[-3., -4.]]),
        );
    }

    #[test]
    fn neg_i32() {
        assert_eq!(ten![1].neg(), ten![-1]);
        assert_eq!(ten!(1, 2, 3).neg(), ten!([-1, -2, -3]));
        assert_eq!(
            ten!([[1, 2], [3, 4]]).neg(), 
            ten!([[-1, -2], [-3, -4]]),
        );
        assert_ne!(
            ten!([[1, 2], [3, 4]]).neg(), 
            ten!([-1, -2, -3, -4]),
        );
        assert_eq!(
            ten!([[1, 2], [3, 4]]).subslice(1, 1).neg(), 
            ten!([[-3, -4]]),
        );
    }

    #[test]
    fn add_f32() {
        assert_eq!(tf32!(2.) + tf32!(3.), tf32!(5.));
        assert_eq!(tf32!(2.) + &tf32!(3.), tf32!(5.));
        assert_eq!(&tf32!(2.) + tf32!(3.), tf32!(5.));
        assert_eq!(&tf32!(2.) + &tf32!(3.), tf32!(5.));

        assert_eq!(tf32!(2.) + 3., tf32!(5.));
        assert_eq!(3. + tf32!(2.), tf32!(5.));

        assert_eq!(&tf32!(2.) + 3., tf32!(5.));
        assert_eq!(3. + &tf32!(2.), tf32!(5.));
    }

    #[test]
    fn add_f32_broadcast() {
        assert_eq!(
            tf32!(2.) + tf32!([1., 2., 3.]), 
            tf32!([3., 4., 5.])
        );
        assert_eq!(
            tf32!([1., 2., 3.]) + tf32!(2.), 
            tf32!([3., 4., 5.])
        );
        assert_eq!(
            tf32!([3., 2., 1.]) + tf32!([1., 2., 3.]), 
            tf32!([4., 4., 4.])
        );
        assert_eq!(
            tf32!([[1., 2.], [3., 4.]]) + tf32!([10., 20.]), 
            tf32!([[11., 22.], [13., 24.]]),
        );
    }

    #[test]
    fn add_i32() {
        assert_eq!(ten!(2) + ten!(3), ten!(5));
        assert_eq!(ten!(2) + &ten!(3), ten!(5));
        assert_eq!(&ten!(2) + ten!(3), ten!(5));
        assert_eq!(&ten!(2) + &ten!(3), ten!(5));

        assert_eq!(ten!(2) + 3, ten!(5));
        assert_eq!(3 + ten!(2), ten!(5));

        assert_eq!(&ten!(2) + 3, ten!(5));
        assert_eq!(3 + &ten!(2), ten!(5));
    }
}