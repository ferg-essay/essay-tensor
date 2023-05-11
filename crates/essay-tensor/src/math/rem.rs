use crate::ops::Binop;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rem;

impl Binop for Rem {
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
