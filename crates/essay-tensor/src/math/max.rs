use crate::ops::BinaryKernel;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Max;

impl BinaryKernel for Max {
    #[inline]
    fn f(&self, x: f32, y: f32) -> f32 {
        x.max(y)
    }

    #[inline]
    fn df_dx(&self, _x: f32, _y: f32) -> f32 {
        todo!()
    }

    #[inline]
    fn df_dy(&self, _x: f32, _y: f32) -> f32 {
        todo!()
    }
}
