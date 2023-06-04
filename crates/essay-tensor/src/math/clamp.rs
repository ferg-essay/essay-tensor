use crate::ops::{BinaryKernel, UnaryKernel};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClampScalar(f32, f32);

impl UnaryKernel<f32> for ClampScalar {
    #[inline]
    fn f(&self, x: f32) -> f32 {
        x.clamp(self.0, self.1)
    }

    #[inline]
    fn df_dx(&self, x: f32) -> f32 {
        if self.0 < x && x < self.1 {
            1.
        } else {
            0.
        }
    }
}

impl ClampScalar {
    pub fn new(min: f32, max: f32) -> Self {
        ClampScalar(min, max)
    }
}
