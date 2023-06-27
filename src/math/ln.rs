use crate::ops::UnaryKernel;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ln;

impl UnaryKernel<f32> for Ln {
    #[inline]
    fn f(&self, value: f32) -> f32 {
        value.ln()
    }

    #[inline]
    fn df_dx(&self, value: f32) -> f32 {
        1. / value
    }
}
