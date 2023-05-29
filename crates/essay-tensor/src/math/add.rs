use crate::{
    ops::{BinaryKernel, UnaryKernel}
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Add;

impl BinaryKernel for Add {
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

    /*
    unsafe fn batch_f(&self, n: usize, a: *const f32, b: *const f32, o: *mut f32) {
        for k in 0..n {
            *o.add(k) = self.f(*a.add(k), *b.add(k));
        }
    }
    */
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
    pub fn new(value: f32) -> Self {
        AddScalar(value)
    }
}

#[cfg(test)]
mod test {
    use crate::{prelude::*, function::{Var, Trainer}};

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

        let module = Trainer::compile((), |()| {
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
}
