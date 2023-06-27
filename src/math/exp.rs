use crate::ops::UnaryKernel;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Exp;

impl UnaryKernel<f32> for Exp {
    #[inline]
    fn f(&self, value: f32) -> f32 {
        value.exp()
    }

    #[inline]
    fn df_dx(&self, value: f32) -> f32 {
        value.exp()
    }
}
