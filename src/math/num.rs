use std::ops;

use num_traits::Signed;

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

impl<T: Type + PartialOrd> PartialOrd for Tensor<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

impl<T: Type + ops::Mul<Output=V> + Clone, V: Type> Tensor<T> {
    #[inline]
    pub fn square(&self) -> Tensor<V> {
        self.map(|a| a.clone() * a.clone())
    }
}

//
// Neg
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
// Not
//

impl<T: ops::Not<Output=T> + Type + Clone> ops::Not for Tensor<T> {
    type Output = Tensor<T>;

    fn not(self) -> Self::Output {
        self.map(|v| !v.clone())
    }
}

impl<T: ops::Not<Output=T> + Type + Clone> ops::Not for &Tensor<T> {
    type Output = Tensor<T>;

    fn not(self) -> Self::Output {
        self.map(|v| !v.clone())
    }
}

//
// Binary operations: Add, Sub, Mul, Div, Rem, BitAnd, BitOr, BitXor, Shl, Shr
//

macro_rules! tensor_ops2 {
    ($op:ident, $fun:ident, $x:ident, $y:ident, $binop:expr, $($sty:ty)*) => {
        impl<T, U, V> ops::$op<Tensor<U>> for Tensor<T>
        where
            T: ops::$op<U, Output=V> + Type + Clone,
            U: Type + Clone,
            V: Type
        {
            type Output = Tensor<V>;
        
            fn $fun(self, rhs: Tensor<U>) -> Self::Output {
                self.map2(&rhs, |$x, $y| $binop)
            }
        }

        impl<T, U, V> ops::$op<Tensor<U>> for &Tensor<T>
        where
            T: ops::$op<U, Output=V> + Type + Clone,
            U: Type + Clone,
            V: Type
        {
            type Output = Tensor<V>;
        
            fn $fun(self, rhs: Tensor<U>) -> Self::Output {
                self.map2(&rhs, |$x, $y| $binop)
            }
        }

        impl<T, U, V> ops::$op<&Tensor<U>> for Tensor<T>
        where
            T: ops::$op<U, Output=V> + Type + Clone,
            U: Type + Clone,
            V: Type
        {
            type Output = Tensor<V>;
        
            fn $fun(self, rhs: &Tensor<U>) -> Self::Output {
                self.map2(&rhs, |$x, $y| $binop)
            }
        }

        impl<T, U, V> ops::$op<&Tensor<U>> for &Tensor<T>
        where
            T: ops::$op<U, Output=V> + Type + Clone,
            U: Type + Clone,
            V: Type
        {
            type Output = Tensor<V>;
        
            fn $fun(self, rhs: &Tensor<U>) -> Self::Output {
                self.map2(&rhs, |$x, $y| $binop)
            }
        }

        impl<T, U, V> ops::$op<U> for Tensor<T>
        where
            T: ops::$op<U, Output=V> + Type + Clone,
            U: Type + Clone,
            V: Type
        {
            type Output = Tensor<V>;
    
            fn $fun(self, $y: U) -> Self::Output {
                self.map(|$x| $binop)
            }
        }

        impl<T, U, V> ops::$op<U> for &Tensor<T>
        where
            T: ops::$op<U, Output=V> + Type + Clone,
            U: Type + Clone,
            V: Type
        {
            type Output = Tensor<V>;
    
            fn $fun(self, $y: U) -> Self::Output {
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
    Add, add, x, y, x.clone() + y.clone(),
    i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize f32 f64
);

tensor_ops2!(
    Sub, sub, x, y, x.clone() - y.clone(),
    i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize f32 f64
);

tensor_ops2!(
    Mul, mul, x, y, x.clone() * y.clone(),
    i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize f32 f64
);

tensor_ops2!(
    Div, div, x, y, x.clone() / y.clone(),
    i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize f32 f64
);

tensor_ops2!(
    Rem, rem, x, y, x.clone() % y.clone(),
    i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize f32 f64
);

tensor_ops2!(
    BitAnd, bitand, x, y, x.clone() & y.clone(),
    bool i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize    
);

tensor_ops2!(
    BitOr, bitor, x, y, x.clone() | y.clone(),
    bool i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize
);

tensor_ops2!(
    BitXor, bitxor, x, y, x.clone() ^ y.clone(),
    bool i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize    
);

tensor_ops2!(
    Shl, shl, x, y, x.clone() << y.clone(), 
    i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize    
);

tensor_ops2!(
    Shr, shr, x, y, x.clone() >> y.clone(), 
    i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize    
);

#[cfg(test)]
mod test {
    use std::ops;

    use crate::{ten, tensor::{scalar, Type}};

    #[test]
    fn abs() {
        assert_eq!(ten!(-1.).abs(), ten!(1.));
        assert_eq!(ten![1., -2., 3.].abs(), ten![1., 2., 3.]);
        assert_eq!(
            ten![[1., -2.], [-3., 4.]].abs(), 
            ten![[1., 2.], [3., 4.]],
        );
        assert_ne!(
            ten![[-1., 2.], [3., -4.]].abs(), 
            ten![1., 2., 3., 4.],
        );
        assert_eq!(
            ten![[1., 2.], [-3., 4.]].subslice(1, 1).abs(), 
            ten![[3., 4.]],
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
        assert_eq!(ten![1., 2., 3.].neg(), ten![-1., -2., -3.]);
        assert_eq!(
            ten![[1., 2.], [3., 4.]].neg(), 
            ten![[-1., -2.], [-3., -4.]],
        );
        assert_ne!(
            ten![[1., 2.], [3., 4.]].neg(), 
            ten![-1., -2., -3., -4.],
        );
        assert_eq!(
            ten![[1., 2.], [3., 4.]].subslice(1, 1).neg(), 
            ten![[-3., -4.]],
        );
    }

    #[test]
    fn neg_i32() {
        assert_eq!(ten![1].neg(), ten![-1]);
        assert_eq!(ten!(1, 2, 3).neg(), ten![-1, -2, -3]);
        assert_eq!(
            ten![[1, 2], [3, 4]].neg(), 
            ten![[-1, -2], [-3, -4]],
        );
        assert_ne!(
            ten![[1, 2], [3, 4]].neg(), 
            ten![-1, -2, -3, -4],
        );
        assert_eq!(
            ten![[1, 2], [3, 4]].subslice(1, 1).neg(), 
            ten![[-3, -4]],
        );
    }

    #[test]
    fn binop_patterns() {
        assert_eq!(ten![2, 20] + ten![3, 30], ten![5, 50]);
        assert_eq!(ten![2, 20] + &ten![3, 30], ten![5, 50]);
        assert_eq!(&ten![2, 20] + ten![3, 30], ten![5, 50]);
        assert_eq!(&ten![2, 20]+ &ten![3, 30], ten![5, 50]);

        assert_eq!(ten![2, 20] + 3, ten![5, 23]);
        assert_eq!(3 + ten![2, 20], ten![5, 23]);

        assert_eq!(&ten![2, 20] + 3, ten![5, 23]);
        assert_eq!(3 + &ten![2, 20], ten![5, 23]);
    }

    #[test]
    fn binop_patterns_order() {
        assert_eq!(ten![3, 30] - ten![2, 20], ten![1, 10]);
        assert_eq!(ten![3, 30] - &ten![2, 20], ten![1, 10]);
        assert_eq!(&ten![3, 30] - ten![2, 20], ten![1, 10]);
        assert_eq!(&ten![3, 30]- &ten![2, 20], ten![1, 10]);

        assert_eq!(ten![3, 30] - 2, ten![1, 28]);
        assert_eq!(2 - ten![3, 30], ten![-1, -28]);

        assert_eq!(&ten![3, 30] - 2, ten![1, 28]);
        assert_eq!(2 - &ten![3, 30], ten![-1, -28]);
    }

    #[test]
    fn binop_asymmetric() {
        assert_eq!(ten![A(2)] + ten![B(3)], ten![C(203)]);
        assert_eq!(ten![A(2)] + &ten![B(3)], ten![C(203)]);
        assert_eq!(&ten![A(2)] + ten![B(3)], ten![C(203)]);
        assert_eq!(&ten![A(2)] + &ten![B(3)], ten![C(203)]);

        assert_eq!(ten![A(2)] + B(3), ten![C(203)]);
        assert_eq!(&ten![A(2)] + B(3), ten![C(203)]);

        assert_eq!(ten![B(2)] + ten![A(3)], ten![D(2003)]);
        assert_eq!(ten![B(2)] + &ten![A(3)], ten![D(2003)]);
        assert_eq!(&ten![B(2)] + ten![A(3)], ten![D(2003)]);
        assert_eq!(&ten![B(2)] + &ten![A(3)], ten![D(2003)]);

        assert_eq!(ten![B(2)] + A(3), ten![D(2003)]);
        assert_eq!(&ten![B(2)] + A(3), ten![D(2003)]);
    }

    #[derive(Clone, Debug)]
    struct A(isize);
    impl Type for A {}

    #[derive(Clone, Debug)]
    struct B(isize);
    impl Type for B {}

    #[derive(Debug, PartialEq)]
    struct C(isize);
    impl Type for C {}

    #[derive(Debug, PartialEq)]
    struct D(isize);
    impl Type for D {}

    impl ops::Add<B> for A {
        type Output=C;
    
        fn add(self, rhs: B) -> Self::Output {
            C(self.0 * 100 + rhs.0)
        }
    }

    impl ops::Add<A> for B {
        type Output=D;
    
        fn add(self, rhs: A) -> Self::Output {
            D(self.0 * 1000 + rhs.0)
        }
    }

    #[test]
    fn add_f32_broadcast() {
        assert_eq!(
            scalar(2.) + ten![1., 2., 3.], 
            ten![3., 4., 5.]
        );
        assert_eq!(
            ten![1., 2., 3.] + scalar(2.), 
            ten![3., 4., 5.]
        );
        assert_eq!(
            ten![3., 2., 1.] + ten![1., 2., 3.], 
            ten![4., 4., 4.]
        );
        assert_eq!(
            ten![[1., 2.], [3., 4.]] + ten![10., 20.], 
            ten![[11., 22.], [13., 24.]],
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

    #[test]
    fn bitand_patterns() {
        assert_eq!(ten![0x3] & ten![0x6], ten![0x2]);
        assert_eq!(ten![0x3] & &ten![0x6], ten![0x2]);
        assert_eq!(&ten![0x3] & ten![0x6], ten![0x2]);
        assert_eq!(&ten![0x3] & &ten![0x6], ten![0x2]);

        assert_eq!(0x3 & ten![0x6], ten![0x2]);
        assert_eq!(0x3 & &ten![0x6], ten![0x2]);
        assert_eq!(ten![0x3] & 0x6, ten![0x2]);
        assert_eq!(&ten![0x3] & 0x6, ten![0x2]);
    }

    #[test]
    fn bitor_patterns() {
        assert_eq!(ten![0x3] | ten![0x6], ten![0x7]);
        assert_eq!(ten![0x3] | &ten![0x6], ten![0x7]);
        assert_eq!(&ten![0x3] | ten![0x6], ten![0x7]);
        assert_eq!(&ten![0x3] | &ten![0x6], ten![0x7]);

        assert_eq!(0x3 | ten![0x6], ten![0x7]);
        assert_eq!(0x3 | &ten![0x6], ten![0x7]);
        assert_eq!(ten![0x3] | 0x6, ten![0x7]);
        assert_eq!(&ten![0x3] | 0x6, ten![0x7]);
    }

    #[test]
    fn bitxor_patterns() {
        assert_eq!(ten![0x3] ^ ten![0x6], ten![0x5]);
        assert_eq!(ten![0x3] ^ &ten![0x6], ten![0x5]);
        assert_eq!(&ten![0x3] ^ ten![0x6], ten![0x5]);
        assert_eq!(&ten![0x3] ^ &ten![0x6], ten![0x5]);

        assert_eq!(0x3 ^ ten![0x6], ten![0x5]);
        assert_eq!(0x3 ^ &ten![0x6], ten![0x5]);
        assert_eq!(ten![0x3] ^ 0x6, ten![0x5]);
        assert_eq!(&ten![0x3] ^ 0x6, ten![0x5]);
    }

    #[test]
    fn shl_patterns() {
        assert_eq!(ten![0x3] << ten![0x2], ten![0xc]);
        assert_eq!(ten![0x3] << &ten![0x2], ten![0xc]);
        assert_eq!(&ten![0x3] << ten![0x2], ten![0xc]);
        assert_eq!(&ten![0x3] << &ten![0x2], ten![0xc]);

        assert_eq!(0x3 << ten![0x2], ten![0xc]);
        assert_eq!(0x3 << &ten![0x2], ten![0xc]);
        assert_eq!(ten![0x3] << 0x2, ten![0xc]);
        assert_eq!(&ten![0x3] << 0x2, ten![0xc]);
    }

    #[test]
    fn shr_patterns() {
        assert_eq!(ten![0xc] >> ten![0x2], ten![0x3]);
        assert_eq!(ten![0xc] >> &ten![0x2], ten![0x3]);
        assert_eq!(&ten![0xc] >> ten![0x2], ten![0x3]);
        assert_eq!(&ten![0xc] >> &ten![0x2], ten![0x3]);

        assert_eq!(0xc >> ten![0x2], ten![0x3]);
        assert_eq!(0xc >> &ten![0x2], ten![0x3]);
        assert_eq!(ten![0xc] >> 0x2, ten![0x3]);
        assert_eq!(&ten![0xc] >> 0x2, ten![0x3]);
    }

    #[test]
    fn not_patterns() {
        assert_eq!(! ten![0x3], ten![-4]);
        assert_eq!(! &ten![0x3], ten![-4]);
    }

    #[test]
    fn partial_ord() {
        assert!(ten![0] < ten![1]);
        assert!(! (ten![1] < ten![0]));
        assert!(ten![0] <= ten![1]);
        assert!(ten![0] <= ten![0]);
        assert!(! (ten![1] <= ten![0]));

        assert!(ten![1] > ten![0]);
        assert!(! (ten![0] > ten![1]));
        assert!(ten![1] >= ten![0]);
        assert!(ten![1] >= ten![1]);
        assert!(! (ten![0] >= ten![1]));
    }
}