use crate::ops::Uop;


#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Abs;

impl Uop<f32> for Abs {
    #[inline]
    fn f(&self, value: f32) -> f32 {
        value.abs()
    }

    #[inline]
    fn df_dx(&self, value: f32) -> f32 {
        if value > 0. { 1. } else { -1. }
    }
}
