use crate::ops::Uop;


#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sin;

impl Uop<f32> for Sin {
    #[inline]
    fn f(&self, value: f32) -> f32 {
        value.sin()
    }

    #[inline]
    fn df_dx(&self, value: f32) -> f32 {
        - value.cos()
    }
}
