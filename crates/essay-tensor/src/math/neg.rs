use crate::ops::Uop;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neg;

impl Uop<f32> for Neg {
    #[inline]
    fn f(&self, value: f32) -> f32 {
        - value
    }

    #[inline]
    fn df_dx(&self, _value: f32) -> f32 {
        - 1.
    }
}
