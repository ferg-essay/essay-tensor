use crate::ops::BinaryKernel;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Powf;

impl BinaryKernel for Powf {
    #[inline]
    fn f(&self, x: f32, y: f32) -> f32 {
        x.powf(y)
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
