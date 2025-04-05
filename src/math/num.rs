use std::ops;

use num_traits::{Num, Signed};

use crate::tensor::{Tensor, Type};

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

//
// neg
//

impl<T: ops::Neg<Output=T> + Type + Clone> ops::Neg for Tensor<T> {
    type Output = Tensor<T>;

    fn neg(self) -> Self::Output {
        self.map(|v| -v.clone())
    }
}

impl<T: ops::Neg<Output=T> + Type + Clone> ops::Neg for &Tensor<T> {
    type Output = Tensor<T>;

    fn neg(self) -> Self::Output {
        self.map(|v| -v.clone())
    }
}

//
// Num binary operations: Add, Sub, Mul, Div, Rem
//

macro_rules! tensor_ops2 {
    ($ty:ident, $($sty:ty)*, $op:ident, $fun:ident, $x:ident, $y:ident, $binop:expr) => {
        impl<T: $ty + Type + Clone> ops::$op<Tensor<T>> for Tensor<T>
        {
            type Output = Tensor<T>;
        
            fn $fun(self, rhs: Tensor<T>) -> Self::Output {
                self.map2(&rhs, |$x, $y| $binop)
            }
        }

        impl<T: $ty + Type + Clone> ops::$op<Tensor<T>> for &Tensor<T> 
        {
            type Output = Tensor<T>;
        
            fn $fun(self, rhs: Tensor<T>) -> Self::Output {
                self.map2(&rhs, |$x, $y| $binop)
            }
        }

        impl<T: $ty + Type + Clone> ops::$op<&Tensor<T>> for Tensor<T>
        {
            type Output = Tensor<T>;
        
            fn $fun(self, rhs: &Tensor<T>) -> Self::Output {
                self.map2(rhs, |$x, $y| $binop)
            }
        }

        impl<T: $ty + Type + Clone> ops::$op<&Tensor<T>> for &Tensor<T> 
        {
            type Output = Tensor<T>;
        
            fn $fun(self, rhs: &Tensor<T>) -> Self::Output {
                self.map2(rhs, |$x, $y| $binop)
            }
        }

        impl<T: $ty + Type + Clone> ops::$op<T> for Tensor<T> {
            type Output = Tensor<T>;
    
            fn $fun(self, $y: T) -> Self::Output {
                self.map(|$x| $binop)
            }
        }

        impl<T: $ty + Type + Clone> ops::$op<T> for &Tensor<T> {
            type Output = Tensor<T>;
    
            fn $fun(self, $y: T) -> Self::Output {
                self.map(|$x| $binop)
            }
        }

        $(
            impl ops::$op<Tensor<$sty>> for $sty
            {
                type Output = Tensor<$sty>;
        
                fn $fun(self, tensor: Tensor<$sty>) -> Self::Output {
                    let $x = self;
                    tensor.map(|$y| $binop)
                }
            }

            impl ops::$op<&Tensor<$sty>> for $sty {
                type Output = Tensor<$sty>;
        
                fn $fun(self, tensor: &Tensor<$sty>) -> Self::Output {
                    let $x = self;
                    tensor.map(|$y| $binop)
                }
            }
        )*
    }
}

tensor_ops2!(
    Num, i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize f32 f64,
    Add, add, x, y, x.clone() + y.clone()
);

tensor_ops2!(
    Num, i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize f32 f64,
    Sub, sub, x, y, x.clone() - y.clone()
);

tensor_ops2!(
    Num, i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize f32 f64,
    Mul, mul, x, y, x.clone() * y.clone()
);

tensor_ops2!(
    Num, i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize f32 f64,
    Div, div, x, y, x.clone() / y.clone()
);

tensor_ops2!(
    Num, i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize f32 f64,
    Rem, rem, x, y, x.clone() % y.clone()
);

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
    fn signed_types() {
        assert_eq!(ten!(-1i8).abs(), ten!(1i8));
        assert_eq!(ten!(-1i16).abs(), ten!(1i16));
        assert_eq!(ten!(-1i32).abs(), ten!(1i32));
        assert_eq!(ten!(-1i64).abs(), ten!(1i64));
        assert_eq!(ten!(-1isize).abs(), ten!(1isize));

        assert_eq!(ten!(-1.0f32).abs(), ten!(1.0f32));
        assert_eq!(ten!(-1.0f64).abs(), ten!(1.0f64));
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
        assert_eq!(ten![2, 20] + ten![3, 30], ten![5, 50]);
        assert_eq!(
            ten![[2, 20], [200, 2000]] + ten![[3, 30], [300, 3000]],
            ten![[5, 50], [500, 5000]]
        );

        assert_eq!(ten!(2) + &ten!(3), ten!(5));
        assert_eq!(&ten!(2) + ten!(3), ten!(5));
        assert_eq!(&ten!(2) + &ten!(3), ten!(5));

        assert_eq!(ten!(2) + 3, ten!(5));
        assert_eq!(3 + ten!(2), ten!(5));

        assert_eq!(&ten!(2) + 3, ten!(5));
        assert_eq!(3 + &ten!(2), ten!(5));
    }
}