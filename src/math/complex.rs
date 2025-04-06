use std::ops::Neg;

use num_complex::Complex;
use num_traits::{Float, Num, Signed};

use crate::tensor::{Tensor, Type};


macro_rules! map {
    ($id: ident) => {
        #[inline]
        pub fn $id(&self) -> Self {
            self.map(|a| a.$id())
        }
    }
}

macro_rules! map_to_re {
    ($id: ident) => {
        #[inline]
        pub fn $id(&self) -> Tensor<T> {
            self.map(|a| a.$id())
        }
    }
}

macro_rules! map2 {
    ($id: ident) => {
        #[inline]
        pub fn $id(&self, b: &Self) -> Self {
            self.map2(b, |a, b| a.$id(b.clone()))
        }
    }
}

macro_rules! map2_type {
    ($id: ident, $ty:ty) => {
        #[inline]
        pub fn $id(&self, b: &Tensor<$ty>) -> Self {
            self.map2(b, |a, b| a.$id(b.clone()))
        }
    }
}

macro_rules! map2_re {
    ($id: ident) => {
        #[inline]
        pub fn $id(&self, b: &Tensor<T>) -> Self {
            self.map2(b, |a, b| a.$id(b.clone()))
        }
    }
}

impl<T> Tensor<Complex<T>>
where
    T: Type + Clone
{
    pub fn re(&self) -> Tensor<T> {
        self.map(|a| a.re.clone())
    }

    pub fn im(&self) -> Tensor<T> {
        self.map(|a| a.im.clone())
    }
}

impl<T: Type + Clone + Num> Tensor<Complex<T>> {
    map_to_re!(norm_sqr);
    map2_re!(scale);
    map2_re!(unscale);
    map2_type!(powu, u32);
}

impl<T: Type + Clone + Num + Neg<Output = T>> Tensor<Complex<T>> {
    map!(conj);
    map!(inv);
    // map2_type!(powi, i32);
}

impl<T: Type + Clone + Signed> Tensor<Complex<T>> {
    map_to_re!(l1_norm);
}

impl<T: Type + Float> Tensor<Complex<T>> {
    map_to_re!(norm);
    map_to_re!(arg);

    #[inline]
    pub fn to_polar(&self) -> Tensor<(T, T)> {
        self.map(|a| a.to_polar())
    }

    #[inline]
    pub fn from_polar(polar: &Tensor<(T, T)>) -> Tensor<Complex<T>> {
        polar.map(|a| Complex::<T>::from_polar(a.0, a.1))
    }

    //map!(exp);
    //map!(ln);
    //map!(sqrt);
    // map2_type!(powf, T);
    // map2_type!(log, T);
    map2!(powc);
    map2_type!(expf, T);
    //map!(sin);
    //map!(cos);
    //map!(tan);
    // map!(asin);
    // map!(acos);
    // map!(atan);
    // map!(sinh);
    // map!(cosh);
    // map!(tanh);
    // map!(asinh);
    // map!(acosh);
    // map!(atanh);
    map!(finv);
    map2!(fdiv);
}

