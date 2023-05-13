use crate::ops::UnaryKernel;


#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Abs;

impl UnaryKernel<f32> for Abs {
    #[inline]
    fn f(&self, x: f32) -> f32 {
        x.abs()
    }

    #[inline]
    fn df_dx(&self, x: f32) -> f32 {
        if x > 0. { 1. } else { -1. }
    }
}
