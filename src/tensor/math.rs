use std::ops;

use super::{Shape, Tensor};
use num_traits::{Float, Num, One, Signed, Zero};

impl<T: Zero + Clone + 'static> Tensor<T> {
    #[inline]
    pub fn zeros(shape: impl Into<Shape>) -> Self {
        Self::init(shape, || Zero::zero())
    }
}

impl<T: One + Clone + 'static> Tensor<T> {
    #[inline]
    pub fn ones(shape: impl Into<Shape>) -> Self {
        Self::init(shape, || One::one())
    }
}

impl<T: Signed + Clone + 'static> Tensor<T> {
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

impl<T: Ord + Clone + 'static> Tensor<T> {
    #[inline]
    pub fn min(&self, rhs: impl Into<Tensor<T>>) -> Self {
        self.map2(&rhs.into(), |a, b| a.clone().min(b.clone()))
    }

    #[inline]
    pub fn max(&self, rhs: impl Into<Tensor<T>>) -> Self {
        self.map2(&rhs.into(), |a, b| a.clone().max(b.clone()))
    }
}

impl<T: Float + Clone + 'static> Tensor<T> {
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
        impl<T: $ty + Clone + 'static> ops::$op<Tensor<T>> for Tensor<T>
        {
            type Output = Tensor<T>;
        
            fn $fun(self, rhs: Tensor<T>) -> Self::Output {
                $binop(&self, &rhs)
            }
        }

        impl<T: $ty + Clone + 'static> ops::$op<Tensor<T>> for &Tensor<T> 
        {
            type Output = Tensor<T>;
        
            fn $fun(self, rhs: Tensor<T>) -> Self::Output {
                $binop(self, &rhs)
            }
        }

        impl<T: $ty + Clone + 'static> ops::$op<&Tensor<T>> for Tensor<T>
        {
            type Output = Tensor<T>;
        
            fn $fun(self, rhs: &Tensor<T>) -> Self::Output {
                $binop(&self, rhs)
            }
        }

        impl<T: $ty + Clone + 'static> ops::$op<&Tensor<T>> for &Tensor<T> 
        {
            type Output = Tensor<T>;
        
            fn $fun(self, rhs: &Tensor<T>) -> Self::Output {
                $binop(self, rhs)
            }
        }

    impl<T: $ty + Clone + 'static> ops::$op<T> for Tensor<T> {
        type Output = Tensor<T>;
    
        fn $fun(self, rhs: T) -> Self::Output {
            $uop_ts(&self, rhs)
        }
    }

    impl<T: $ty + Clone + 'static> ops::$op<T> for &Tensor<T> {
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

fn add_f32<T: Num + Clone + 'static>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
    a.map2(b, |a, b| a.clone() + b.clone())
}

fn add_f32_ts<T: Num + Clone + 'static>(a: &Tensor<T>, b: T) -> Tensor<T> {
    a.map(|a| a.clone() + b.clone())
}

fn add_f32_st<T: Num + Clone + 'static>(a: T, b: &Tensor<T>) -> Tensor<T> {
    b.map(|b| a.clone() + b.clone())
}

tensor_ops2!(Num, Add, add, add_f32, add_f32_st, add_f32_ts);
tensor_ops2_scalar!(f32, Add, add, add_f32, add_f32_st, add_f32_ts);
tensor_ops2_scalar!(i32, Add, add, add_32, add_f32_st, add_f32_ts);


#[cfg(test)]
mod test {
    use crate::{tensor, tf32};

    #[test]
    fn abs() {
        assert_eq!(tensor!(-1.).abs(), tensor!(1.));
        assert_eq!(tensor!([1., -2., 3.]).abs(), tensor!([1., 2., 3.]));
        assert_eq!(
            tensor!([[1., -2.], [-3., 4.]]).abs(), 
            tensor!([[1., 2.], [3., 4.]]),
        );
        assert_ne!(
            tensor!([[-1., 2.], [3., -4.]]).abs(), 
            tensor!([1., 2., 3., 4.]),
        );
        assert_eq!(
            tensor!([[1., 2.], [-3., 4.]]).subslice(1, 1).abs(), 
            tensor!([[3., 4.]]),
        );
    }

    #[test]
    fn abs_i32() {
        assert_eq!(tensor!(-1i32).abs(), tensor!(1i32));
        assert_eq!(tensor!([1, -2, 3]).abs(), tensor!([1, 2, 3]));
        assert_eq!(
            tensor!([[1, -2], [-3, 4]]).abs(), 
            tensor!([[1, 2], [3, 4]]),
        );
        assert_ne!(
            tensor!([[-1, 2], [3, -4]]).abs(), 
            tensor!([1, 2, 3, 4]),
        );
        assert_eq!(
            tensor!([[1, 2], [-3, 4]]).subslice(1, 1).abs(), 
            tensor!([[3, 4]]),
        );
    }

    #[test]
    fn neg() {
        assert_eq!(tensor!(1.).neg(), tensor!(-1.));
        assert_eq!(tensor!([1., 2., 3.]).neg(), tensor!([-1., -2., -3.]));
        assert_eq!(
            tensor!([[1., 2.], [3., 4.]]).neg(), 
            tensor!([[-1., -2.], [-3., -4.]]),
        );
        assert_ne!(
            tensor!([[1., 2.], [3., 4.]]).neg(), 
            tensor!([-1., -2., -3., -4.]),
        );
        assert_eq!(
            tensor!([[1., 2.], [3., 4.]]).subslice(1, 1).neg(), 
            tensor!([[-3., -4.]]),
        );
    }

    #[test]
    fn neg_i32() {
        assert_eq!(tensor!(1).neg(), tensor!(-1));
        assert_eq!(tensor!([1, 2, 3]).neg(), tensor!([-1, -2, -3]));
        assert_eq!(
            tensor!([[1, 2], [3, 4]]).neg(), 
            tensor!([[-1, -2], [-3, -4]]),
        );
        assert_ne!(
            tensor!([[1, 2], [3, 4]]).neg(), 
            tensor!([-1, -2, -3, -4]),
        );
        assert_eq!(
            tensor!([[1, 2], [3, 4]]).subslice(1, 1).neg(), 
            tensor!([[-3, -4]]),
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
        assert_eq!(tensor!(2) + tensor!(3), tensor!(5));
        assert_eq!(tensor!(2) + &tensor!(3), tensor!(5));
        assert_eq!(&tensor!(2) + tensor!(3), tensor!(5));
        assert_eq!(&tensor!(2) + &tensor!(3), tensor!(5));

        assert_eq!(tensor!(2) + 3, tensor!(5));
        assert_eq!(3 + tensor!(2), tensor!(5));

        assert_eq!(&tensor!(2) + 3, tensor!(5));
        assert_eq!(3 + &tensor!(2), tensor!(5));
    }
}