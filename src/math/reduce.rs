use std::ops;

use num_traits::{Float, Zero};

use crate::tensor::{Axis, Tensor, TensorData, Type};

use super::reduce_std;

pub fn reduce_mean(a: &Tensor) -> Tensor {
    // reduce_op(a, ReduceMean, opt)
    todo!();
}

impl Tensor {
    pub fn reduce_mean(&self) -> Tensor {
        self.fold_into(Mean::default(), |s, v| s.update(*v))
    }

    pub fn reduce_std(&self) -> Tensor {
        reduce_std(&self)
    }
    
    pub fn reduce_var(&self) -> Tensor {
        // reduce_std(self, opt)
        todo!();
    }

    pub fn reduce_min(&self) -> Tensor {
        // reduce_min(self)
        todo!();
    }

    pub fn reduce_min_opt(&self) -> Tensor {
        // reduce_min_opt(self, opt)
        todo!();
    }

    pub fn reduce_max(&self) -> Tensor {
        // self.fold(T::max_value(), |s, v| s.min(v))
        todo!()
    }

    pub fn reduce_max_opt(&self) -> Tensor {
        // reduce_min_opt(self, opt)
        todo!();
    }
}

impl<T: Type + ops::Add<Output=T> + Clone> Tensor<T> {
    pub fn reduce_sum(&self) -> Tensor<T> {
        self.reduce(|s, v| s + v)
    }
    pub fn reduce_sum_axis(&self, axis: impl Into<Axis>) -> Tensor<T> {
        self.reduce_axis(axis, |s, v| s + v.clone())
    }
}

impl<T: Type + ops::Mul<Output=T> + Clone> Tensor<T> {
    pub fn reduce_product(&self) -> Tensor<T> {
        self.reduce(|s, v| s * v)
    }

    pub fn reduce_product_axis(&self, axis: impl Into<Axis>) -> Tensor<T> {
        self.reduce_axis(axis, |s, v| s * v)
    }
}

/*
pub fn reduce<T, U, S, F>(
    tensor: &Tensor<T>,
    init: S,
    f: F,
) -> Tensor<U> 
where
    U: 'static,
    S: Clone + Into<U>,
    F: FnMut(S, &T) -> S
{
    // tensor.reduce(init, f)
    todo!()
} 
*/   
/*
pub fn reduce_axis<T, U, S, F>(
    tensor: &Tensor<T>,
    axis: impl Into<Axis>,
    init: S,
    mut f: F,
) -> Tensor<U> 
where
    T: Type,
    U: Type,
    S: Clone + Into<U>,
    F: FnMut(S, &T) -> S,
{
    let axis = axis.into();

    let (o_shape, batch, a_len, inner) = axis.reduce(tensor.shape());

    unsafe {
        TensorData::<U>::unsafe_init(o_shape.size(), |o| {
            let a = tensor.as_slice();

            for n in 0..batch {
                for i in 0..inner {
                    let mut state = init.clone();

                    for k in 0..a_len {
                        let v = &a[n * a_len * inner + i + k * inner];

                        state = (f)(state, v);
                    }

                    o.add(n * inner + i).write(state.into());
                }
            }
        }).into_tensor(o_shape)
    }
}
*/

#[derive(Default, Clone, Debug)]
struct Mean {
    n: usize,
    sum: f32,
}

impl Mean {
    fn update(self, value: f32) -> Self {
        Self {
            n: self.n + 1,
            sum: self.sum + value,
        }
    }
}

impl From<Mean> for f32 {
    fn from(value: Mean) -> Self {
        value.sum / value.n.max(1) as f32
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn reduce_axis() {
        assert_eq!(
            ten![[1, 2], [3, 4]].reduce_axis(None, |s, v| s + v), 
            ten![10]
        );
        assert_eq!(
            ten![[1, 2], [3, 4]].reduce_axis(1, |s, v| s + v), 
            ten![3, 7]
        );
        assert_eq!(
            ten![[1, 2], [3, 4]].reduce_axis(0, |s, v| s + v), 
            ten![4, 6]
        );
    }

    #[test]
    fn reduce_sum_n() {
        assert_eq!(tf32!([1.]).reduce_sum(), tf32!(1.));
        assert_eq!(tf32!([1., 10.]).reduce_sum(), tf32!(11.));
        assert_eq!(tf32!([10., 1.]).reduce_sum(), tf32!(11.));
    }

    #[test]
    fn reduce_sum_1xn() {
        assert_eq!(tf32!([[1.]]).reduce_sum(), tf32!([1.]));
        assert_eq!(tf32!([[1., 10.]]).reduce_sum(), tf32!([11.]));
        assert_eq!(tf32!([[10., 1.]]).reduce_sum(), tf32!([11.]));
    }

    #[test]
    fn reduce_sum_2xn() {
        assert_eq!(tf32!([[1.], [2.]]).reduce_sum(), tf32!([1., 2.]));
        assert_eq!(tf32!([[1., 10.], [2., 20.]]).reduce_sum(), tf32!([11., 22.]));
        assert_eq!(tf32!([[20., 2.], [10., 1.]]).reduce_sum(), tf32!([22., 11.]));
    }

    #[test]
    fn reduce_sum_2x1xn() {
        assert_eq!(tf32!([[[1.]], [[2.]]]).reduce_sum(), tf32!([[1.], [2.]]));
        assert_eq!(tf32!([[[1., 10.]], [[2., 20.]]]).reduce_sum(), tf32!([[11.], [22.]]));
        assert_eq!(tf32!([[[20., 2.]], [[10., 1.]]]).reduce_sum(), tf32!([[22.], [11.]]));
    }

    #[test]
    fn reduce_sum_1xn_axis_none() {
        assert_eq!(tf32!([[1.]]).reduce_sum_axis(None), tf32!(1.));
        assert_eq!(tf32!([[1., 10.]]).reduce_sum_axis(None), tf32!(11.));
        assert_eq!(tf32!([[10., 1.]]).reduce_sum_axis(None), tf32!(11.));
    }

    #[test]
    fn reduce_sum_2xn_axis_none() {
        assert_eq!(tf32!([[1.], [2.]]).reduce_sum_axis(None), tf32!(3.));
        assert_eq!(tf32!([[1., 10.], [100., 1000.]]).reduce_sum_axis(None), tf32!(1111.));
    }

    #[test]
    fn reduce_sum_2x1x1xn_axis_none() {
        assert_eq!(tf32!([[[[1.]]], [[[2.]]]]).reduce_sum_axis(None), tf32!(3.));
        assert_eq!(tf32!([[[[1., 10.]]], [[[100., 1000.]]]]).reduce_sum_axis(None), tf32!(1111.));
    }

    #[test]
    fn reduce_sum_2xn_axis_0() {
        assert_eq!(tf32!([[1.], [2.]]).reduce_sum_axis(0), tf32!([3.]));
        assert_eq!(tf32!([[1., 10.], [100., 1000.]]).reduce_sum_axis(0), tf32!([101., 1010.]));
    }
    
    #[test]
    fn reduce_mean() {
        assert_eq!(tf32!([1.]).reduce_mean(), tf32!([1.]));
        assert_eq!(tf32!([1., 3.]).reduce_mean(), tf32!([2.]));
        assert_eq!(tf32!([[1., 3.], [4., 0.]]).reduce_mean(), tf32!([2., 4.]));
    }

    /*
    #[test]
    fn reduce_mean_axis_m1() {
        assert_eq!(tf32!([1.]).reduce_mean(().axis(Some(-1))), tf32!(1.));
        assert_eq!(tf32!([1., 3.]).reduce_mean(().axis(Some(-1))), tf32!(2.));
        assert_eq!(tf32!([[1., 3.], [4., 6.]]).reduce_mean(().axis(Some(-1))), tf32!([2., 5.]));
        assert_eq!(tf32!([[[1., 3.]], [[4., 6.]]]).reduce_mean(().axis(Some(-1))), tf32!([[2.], [5.]]));
    }
    
    #[test]
    fn reduce_mean_axis_0() {
        assert_eq!(tf32!([1.]).reduce_mean(().axis(Some(0))), tf32!(1.));
        assert_eq!(tf32!([1., 3.]).reduce_mean(().axis(Some(0))), tf32!(2.));
        assert_eq!(tf32!([[1., 3.], [4., 6.]]).reduce_mean(().axis(Some(0))), tf32!([2.5, 4.5]));
        assert_eq!(tf32!([[[1., 3.]], [[4., 6.]]]).reduce_mean(().axis(Some(0))), tf32!([[2.5, 4.5]]));
    }
    */
}
