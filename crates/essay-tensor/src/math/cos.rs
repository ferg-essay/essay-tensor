use crate::ops::Uop;


#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cos;

impl Uop<f32> for Cos {
    #[inline]
    fn f(&self, value: f32) -> f32 {
        value.cos()
    }

    #[inline]
    fn df_dx(&self, value: f32) -> f32 {
        value.sin()
    }
}