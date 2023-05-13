use crate::ops::UnaryKernel;


#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cos;

impl UnaryKernel<f32> for Cos {
    #[inline]
    fn f(&self, x: f32) -> f32 {
        x.cos()
    }

    #[inline]
    fn df_dx(&self, x: f32) -> f32 {
        - x.sin()
    }
}