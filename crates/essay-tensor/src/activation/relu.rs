use crate::{
    ops::Uop
};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct ReLU;

impl Uop<f32> for ReLU {
    #[inline]
    fn f(&self, value: f32) -> f32 {
        value.max(0.)
    }

    fn df_dx(&self, value: f32) -> f32 {
        if value >= 0. { 1. } else { 0. }
    }
}
