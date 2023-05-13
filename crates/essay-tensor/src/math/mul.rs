use crate::ops::BinaryKernel;

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
