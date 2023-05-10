use crate::{
    ops::Binop
};


#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Add;

impl Binop for Add {
    #[inline]
    fn f(&self, x: f32, y: f32) -> f32 {
        x + y
    }

    #[inline]
    fn df_dx(&self, _x: f32, _y: f32) -> f32 {
        1.
    }

    #[inline]
    fn df_dy(&self, _x: f32, _y: f32) -> f32 {
        1.
    }
}
