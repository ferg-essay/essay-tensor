use num_traits::Float;

use crate::tensor::{Tensor, Type};


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
    pub fn mul_add(&self, b: &Tensor<T>, c: &Tensor<T>) -> Tensor<T> {
        self.map3(b, c, |a, b, c| a.mul_add(b.clone(), c.clone()))
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

    #[inline]
    pub fn cbrt(&self) -> Tensor<T> {
        self.map(|a| a.cbrt())
    }

    #[inline]
    pub fn hypot(&self, b: &Tensor<T>) -> Tensor<T> {
        self.map2(b, |a, b| a.hypot(b.clone()))
    }

    #[inline]
    pub fn sin(&self) -> Tensor<T> {
        self.map(|a| a.sin())
    }

    #[inline]
    pub fn asin(&self) -> Tensor<T> {
        self.map(|a| a.asin())
    }

    #[inline]
    pub fn sinh(&self) -> Tensor<T> {
        self.map(|a| a.sinh())
    }

    #[inline]
    pub fn asinh(&self) -> Tensor<T> {
        self.map(|a| a.asinh())
    }

    #[inline]
    pub fn cos(&self) -> Tensor<T> {
        self.map(|a| a.cos())
    }

    #[inline]
    pub fn acos(&self) -> Tensor<T> {
        self.map(|a| a.acos())
    }

    #[inline]
    pub fn cosh(&self) -> Tensor<T> {
        self.map(|a| a.cos())
    }

    #[inline]
    pub fn acosh(&self) -> Tensor<T> {
        self.map(|a| a.acosh())
    }

    #[inline]
    pub fn sin_cos(&self) -> Tensor<(T, T)> {
        self.map(|a| a.sin_cos())
    }

    #[inline]
    pub fn tan(&self) -> Tensor<T> {
        self.map(|a| a.tan())
    }

    #[inline]
    pub fn atan(&self) -> Tensor<T> {
        self.map(|a| a.atan())
    }

    #[inline]
    pub fn atan2(&self, b: &Tensor<T>) -> Tensor<T> {
        self.map2(b, |a, b| a.atan2(b.clone()))
    }

    #[inline]
    pub fn tanh(&self) -> Tensor<T> {
        self.map(|a| a.tanh())
    }

    #[inline]
    pub fn atanh(&self) -> Tensor<T> {
        self.map(|a| a.atanh())
    }

    #[inline]
    pub fn exp_m1(&self) -> Tensor<T> {
        self.map(|a| a.exp_m1())
    }

    #[inline]
    pub fn ln_1p(&self) -> Tensor<T> {
        self.map(|a| a.ln_1p())
    }

    #[inline]
    pub fn copysign(&self, sign: &Tensor<T>) -> Tensor<T> {
        self.map2(sign, |a, b| a.copysign(b.clone()))
    }
}

#[cfg(test)]
mod test {
    use std::f32::consts::PI;

    use crate::ten;

    #[test]
    fn mul_add() {
        assert_eq!(ten![2.].mul_add(&ten![3.], &ten![10.]), ten![(16.)]);
    }

    #[test]
    fn sin_cos() {
        assert_eq!(ten![0.].sin_cos(), ten![(0., 1.)]);
        assert_eq!(
            ten![0., PI / 2., PI, 3. * PI / 2., 2. * PI].sin_cos(), 
            ten![
                (0.0, 1.0),
                (1.0, -4.371139e-8),
                (-8.742278e-8, -1.0),
                (-1.0, 1.1924881e-8),
                (1.7484555e-7, 1.0)
            ]
        );
    }

}