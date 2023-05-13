use crate::ops::UnaryKernel;


#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sin;

impl UnaryKernel<f32> for Sin {
    #[inline]
    fn f(&self, x: f32) -> f32 {
        x.sin()
    }

    #[inline]
    fn df_dx(&self, x: f32) -> f32 {
        x.cos()
    }
}
