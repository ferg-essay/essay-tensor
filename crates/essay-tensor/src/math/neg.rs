use crate::ops::UnaryKernel;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neg;

impl UnaryKernel<f32> for Neg {
    #[inline]
    fn f(&self, value: f32) -> f32 {
        - value
    }

    #[inline]
    fn df_dx(&self, _value: f32) -> f32 {
        - 1.
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