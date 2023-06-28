use crate::ops::{BinaryKernel, UnaryKernel};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sub;

impl BinaryKernel for Sub {
    #[inline]
    fn f(&self, x: &f32, y: &f32) -> f32 {
        x - y
    }

    #[inline]
    fn df_dx(&self, _x: f32, _y: f32) -> f32 {
        1.
    }

    #[inline]
    fn df_dy(&self, _x: f32, _y: f32) -> f32 {
        -1.
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubST(f32);

impl UnaryKernel<f32> for SubST {
    #[inline]
    fn f(&self, x: f32) -> f32 {
        self.0 - x
    }

    #[inline]
    fn df_dx(&self, _x: f32) -> f32 {
        - 1.
    }
}

impl SubST {
    pub fn new(value: f32) -> Self {
        SubST(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubTS(f32);

impl UnaryKernel<f32> for SubTS {
    #[inline]
    fn f(&self, x: f32) -> f32 {
        x - self.0
    }

    #[inline]
    fn df_dx(&self, _x: f32) -> f32 {
        1.
    }
}

impl SubTS {
    pub fn new(value: f32) -> Self {
        SubTS(value)
    }
}
