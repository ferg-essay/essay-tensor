use crate::{
    ops::{Uop}
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Abs;

impl Uop<f32> for Abs {
    #[inline]
    fn f(&self, value: f32) -> f32 {
        value.abs()
    }

    #[inline]
    fn df_dx(&self, value: f32) -> f32 {
        if value > 0. { 1. } else { -1. }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cos;

impl Uop<f32> for Cos {
    #[inline]
    fn f(&self, value: f32) -> f32 {
        value.cos()
    }

    #[inline]
    fn df_dx(&self, value: f32) -> f32 {
        value.sin()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Exp;

impl Uop<f32> for Exp {
    #[inline]
    fn f(&self, value: f32) -> f32 {
        value.exp()
    }

    #[inline]
    fn df_dx(&self, value: f32) -> f32 {
        value.exp()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ln;

impl Uop<f32> for Ln {
    #[inline]
    fn f(&self, value: f32) -> f32 {
        value.ln()
    }

    #[inline]
    fn df_dx(&self, value: f32) -> f32 {
        1. / value
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neg;

impl Uop<f32> for Neg {
    #[inline]
    fn f(&self, value: f32) -> f32 {
        - value
    }

    #[inline]
    fn df_dx(&self, _value: f32) -> f32 {
        - 1.
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sin;

impl Uop<f32> for Sin {
    #[inline]
    fn f(&self, value: f32) -> f32 {
        value.sin()
    }

    #[inline]
    fn df_dx(&self, value: f32) -> f32 {
        - value.cos()
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::{*};

    #[test]
    fn neg_f() {
        assert_eq!(- tensor!(1.), tensor!(-1.));
        assert_eq!(- tensor!([1.]), tensor!([-1.]));
        assert_eq!(- tensor!([[1.]]), tensor!([[-1.]]));
        assert_eq!(- tensor!([[[1.]]]), tensor!([[[-1.]]]));
        assert_eq!(- tensor!([[[[1.]]]]), tensor!([[[[-1.]]]]));

        assert_eq!(- tensor!([1., 2.]), tensor!([-1., -2.]));
        assert_eq!(- tensor!([[-1., -2.], [-3., -4.]]), 
            tensor!([[1., 2.], [3., 4.]]));
    }
}