use crate::{
    ops::UnaryKernel
};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct SoftPlus;

impl UnaryKernel<f32> for SoftPlus {
    #[inline]
    fn f(&self, value: f32) -> f32 {
        (value.exp() + 1.).ln()
    }

    fn df_dx(&self, _value: f32) -> f32 {
        todo!()
    }
}

