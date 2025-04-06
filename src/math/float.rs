use num_traits::{Float, MulAdd};

use crate::tensor::{Tensor, Type};

macro_rules! map {
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
        pub fn $id(&self, b: &Tensor<T>) -> Tensor<T> {
            self.map2(b, |a, b| a.$id(b.clone()))
        }
    }
}

impl<T: Type + Float + Clone> Tensor<T> {
    map!(floor);
    map!(ceil);
    map!(round);
    map!(trunc);
    map!(fract);
    map!(signum);
    map!(recip);

    #[inline]
    pub fn powi(&self, rhs: impl Into<Tensor<i32>>) -> Tensor<T> {
        self.map2(&rhs.into(), |a, b| a.powi(*b))
    }

    map2!(powf);

    map!(sqrt);
    map!(exp);
    map!(exp2);
    map!(ln);
    map2!(log);
    map!(log2);
    map!(log10);
    map!(to_degrees);
    map!(to_radians);
    map!(cbrt);
    map2!(hypot);
    map!(sin);
    map!(asin);
    map!(sinh);
    map!(asinh);
    map!(cos);
    map!(acos);
    map!(cosh);
    map!(acosh);

    #[inline]
    pub fn sin_cos(&self) -> Tensor<(T, T)> {
        self.map(|a| a.sin_cos())
    }

    map!(tan);
    map!(atan);
    map!(tanh);
    map!(atanh);
    map2!(atan2);
    map!(exp_m1);
    map!(ln_1p);
    map2!(copysign);
}

impl<T: Type + MulAdd<Output=T> + Clone> MulAdd<&Tensor<T>, &Tensor<T>> for Tensor<T> {
    type Output = Tensor<T>;

    fn mul_add(self, b: &Tensor<T>, c: &Tensor<T>) -> Self::Output {
        self.map3(b, c, |a, b, c| a.clone().mul_add(b.clone(), c.clone()))
    }
}

#[cfg(test)]
mod test {
    use std::f32::consts::PI;
    use num_traits::MulAdd;

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