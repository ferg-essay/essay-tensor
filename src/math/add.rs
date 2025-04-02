use std::{marker::PhantomData, ops};

use crate::{
    ops::{BinaryKernel, UnaryKernel}, tensor::Dtype,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Add;

impl BinaryKernel<f32> for Add {
    #[inline]
    fn f(&self, x: &f32, y: &f32) -> f32 {
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AddScalar(f32);

impl UnaryKernel<f32> for AddScalar {
    #[inline]
    fn f(&self, x: f32) -> f32 {
        x + self.0
    }

    #[inline]
    fn df_dx(&self, _x: f32) -> f32 {
        1.
    }
}

impl AddScalar {
    pub fn _new(value: f32) -> Self {
        AddScalar(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Add2<D: ops::Add> {
    marker: PhantomData<D>,
}

impl<D: ops::Add> Add2<D> {
    pub(crate) fn _new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

impl<D: Dtype + ops::Add<Output=D> + Copy> BinaryKernel<D> for Add2<D> {
    #[inline]
    fn f(&self, x: &D, y: &D) -> D {
        *x + *y
    }

    #[inline]
    fn df_dx(&self, _x: D, _y: D) -> D {
        todo!() // D::one()
    }

    #[inline]
    fn df_dy(&self, _x: D, _y: D) -> D {
        todo!() // D::one()
    }
}

#[cfg(test)]
mod test {
    use crate::{prelude::*, model::{Var, Trainer}};

    #[test]
    fn test_add() {
        assert_eq!(tf32!(2.) + tf32!(3.), tf32!(5.));
        assert_eq!(tf32!([3., 4.]) + tf32!([1., 2.]), tf32!([4., 6.]));
    }

    #[test]
    fn test_add_into() {
        assert_eq!(tensor!(2.) + tensor!(3.), tensor!(5.));
        assert_eq!(tensor!([3., 4.]) + tensor!([1., 2.]), tensor!([4., 6.]));
    }

    #[test]
    fn test_add_df() {
        let x = Var::new("x", tensor!([1., 2.]));
        let y = Var::new("y", tensor!([3., 4.]));

        let module = Trainer::compile((), |(), _| {
            &x + &y
        }); // .training(&[&x, &y]);
        let train = module.train(());

        assert_eq!(train.value(), tensor!([4., 6.]));
        assert_eq!(train.gradient(&x), tensor!([1., 1.]));
        assert_eq!(train.gradient(&y), tensor!([1., 1.]));
    }

    #[test]
    fn op_tensor_x_f32() {
        assert_eq!(tf32!(2.) + tf32!(3.), tf32!(5.));
        assert_eq!(tf32!(2.) + &tf32!(3.), tf32!(5.));
        assert_eq!(&tf32!(2.) + tf32!(3.), tf32!(5.));
        assert_eq!(&tf32!(2.) + &tf32!(3.), tf32!(5.));

        assert_eq!(tf32!(2.) + 3., tf32!(5.));
        assert_eq!(3. + tf32!(2.), tf32!(5.));

        assert_eq!(&tf32!(2.) + 3., tf32!(5.));
        assert_eq!(3. + &tf32!(2.), tf32!(5.));
    }

    /*
    #[test]
    fn add_c32() {
        assert_eq!(
            add(tc32!([(1., 10.)]), tc32!([(2., 20.)])),
            tc32!([(3., 30.)]),
        );
        assert_eq!(
            tc32!([(1., 10.)]) + tc32!([(2., 20.)]),
            tc32!([(3., 30.)]),
        );
    }
    */
}
