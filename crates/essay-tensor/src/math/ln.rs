use crate::ops::Uop;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ln;

impl Uop<f32> for Ln {
    #[inline]
    fn f(&self, value: f32) -> f32 {
        value.ln()
    }

    #[inline]
    fn df_dx(&self, value: f32) -> f32 {
        1. / value
    }
}
