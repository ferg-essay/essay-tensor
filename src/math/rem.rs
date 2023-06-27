use crate::ops::{BinaryKernel, UnaryKernel};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rem;

impl BinaryKernel for Rem {
    #[inline]
    fn f(&self, x: f32, y: f32) -> f32 {
        x % y
    }

    #[inline]
    fn df_dx(&self, _x: f32, _y: f32) -> f32 {
        unimplemented!();
    }

    #[inline]
    fn df_dy(&self, _x: f32, _y: f32) -> f32 {
        unimplemented!();
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RemST(f32);

impl UnaryKernel<f32> for RemST {
    #[inline]
    fn f(&self, x: f32) -> f32 {
        self.0 % x
    }

    #[inline]
    fn df_dx(&self, x: f32) -> f32 {
        - x.powi(2).recip()
    }
}

impl RemST {
    pub fn new(value: f32) -> Self {
        RemST(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RemTS(f32);

impl UnaryKernel<f32> for RemTS {
    #[inline]
    fn f(&self, x: f32) -> f32 {
        x / self.0
    }

    #[inline]
    fn df_dx(&self, _x: f32) -> f32 {
        self.0.recip()
    }
}

impl RemTS {
    pub fn new(value: f32) -> Self {
        RemTS(value)
    }
}
