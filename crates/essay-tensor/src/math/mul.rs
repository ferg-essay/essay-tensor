use crate::ops::{BinaryKernel, UnaryKernel};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mul;

impl BinaryKernel for Mul {
    #[inline]
    fn f(&self, x: f32, y: f32) -> f32 {
        x * y
    }

    #[inline]
    fn df_dx(&self, _x: f32, y: f32) -> f32 {
        y
    }

    #[inline]
    fn df_dy(&self, x: f32, _y: f32) -> f32 {
        x
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MulScalar(f32);

impl UnaryKernel<f32> for MulScalar {
    #[inline]
    fn f(&self, x: f32) -> f32 {
        x * self.0
    }

    #[inline]
    fn df_dx(&self, x: f32) -> f32 {
        x
    }
}

impl MulScalar {
    pub fn new(value: f32) -> Self {
        MulScalar(value)
    }
}
