use crate::ops::{BinaryKernel, UnaryKernel};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Div;

impl BinaryKernel for Div {
    #[inline]
    fn f(&self, x: f32, y: f32) -> f32 {
        x / y
    }

    #[inline]
    fn df_dx(&self, _x: f32, y: f32) -> f32 {
        y.recip()
    }

    #[inline]
    fn df_dy(&self, x: f32, y: f32) -> f32 {
        - x / y.powi(2)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DivST(f32);

impl UnaryKernel<f32> for DivST {
    #[inline]
    fn f(&self, x: f32) -> f32 {
        self.0 / x
    }

    #[inline]
    fn df_dx(&self, x: f32) -> f32 {
        - x.powi(2).recip()
    }
}

impl DivST {
    pub fn new(value: f32) -> Self {
        DivST(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DivTS(f32);

impl UnaryKernel<f32> for DivTS {
    #[inline]
    fn f(&self, x: f32) -> f32 {
        x / self.0
    }

    #[inline]
    fn df_dx(&self, _x: f32) -> f32 {
        self.0.recip()
    }
}

impl DivTS {
    pub fn new(value: f32) -> Self {
        DivTS(value)
    }
}
