use std::ops;

use num_traits::{Float, Zero};

use crate::tensor::{Axis, FoldState, Tensor, Type};

impl<T: Type + Float + Zero> Tensor<T> {
    pub fn reduce_hypot(&self) -> Tensor<T> {
        self.fold(Hypot::default(), |s, v| s.update(v.clone()))
    }
}

impl<T: Type + Float> Tensor<T> {
    pub fn reduce_min(&self) -> Self {
        self.reduce(|s, v| s.min(v))
    }

    pub fn reduce_min_axis(&self, axis: impl Into<Axis>) -> Self {
        self.reduce_axis(axis, |s, v| s.min(v))
    }

    pub fn reduce_max(&self) -> Self {
        self.reduce(|s, v| s.max(v))
    }

    pub fn reduce_max_axis(&self, axis: impl Into<Axis>) -> Self {
        self.reduce_axis(axis, |s, v| s.max(v))
    }
}

impl Tensor<f32> {
    pub fn reduce_mean(&self) -> Tensor {
        self.fold(Mean::default(), |s, v| s.update(*v))
    }

    pub fn reduce_mean_axis(&self, axis: impl Into<Axis>) -> Tensor {
        self.fold_axis(axis, Mean::default(), |s, v| s.update(*v))
    }

    pub fn reduce_std(&self) -> Tensor {
        self.fold(Std::default(), |s, v| s.update(*v))
    }
    
    pub fn reduce_std_axis(&self, axis: impl Into<Axis>) -> Tensor {
        self.fold_axis(axis, Std::default(), |s, v| s.update(*v))
    }
    
    pub fn reduce_variance(&self) -> Tensor {
        self.fold(Var::default(), |s, v| s.update(*v))
    }
    
    pub fn reduce_variance_axis(&self, axis: impl Into<Axis>) -> Tensor {
        self.fold_axis(axis, Var::default(), |s, v| s.update(*v))
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

#[derive(Clone, Debug)]
struct Hypot<T: Float>(T);

impl<T: Float> Default for Hypot<T> {
    fn default() -> Self {
        Self(T::zero())
    }
}

impl<T: Float> Hypot<T> {
    fn update(self, v: T) -> Self {
        Hypot(self.0 + v * v)
    }
}

impl<T: Float> FoldState for Hypot<T> {
    type Out = T;

    fn into_result(self) -> Self::Out {
        self.0.sqrt()
    }
}

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

impl FoldState for Mean {
    type Out = f32;

    fn into_result(self) -> Self::Out {
        self.sum / self.n.max(1) as f32
    }
}

#[derive(Clone)]
struct Std {
    k: usize,
    m: f32,
    s: f32,
}

impl Std {
    fn update(self, x: f32) -> Self {
        // from Welford 1962
        if self.k == 0 {
            Std {
                k: 1,
                m: x,
                s: 0.,
            }
        } else {
            let k = self.k + 1;
            let m = self.m + (x - self.m) / k as f32;

            Std {
                k,
                m, 
                s: self.s + (x - self.m) * (x - m),
            }
        }
    }
}

impl Default for Std {
    fn default() -> Self {
        Self { 
            k: 0,
            s: 0.,
            m: 0.,
        }
    }
}

impl FoldState for Std {
    type Out=f32;

    fn into_result(self) -> Self::Out {
        if self.k > 1 { 
            (self.s / self.k as f32).sqrt()
        } else {
            0.
        }
    }
}

#[derive(Clone, Default)]
struct Var {
    k: usize,
    m: f32,
    s: f32,
}

impl Var {
    fn update(self, x: f32) -> Var {
        // from Welford 1962
        if self.k == 0 {
            Var {
                k: 1,
                m: x,
                s: 0.,
            }
        } else {
            let k = self.k + 1;
            let m = self.m + (x - self.m) / k as f32;

            Var {
                k,
                m, 
                s: self.s + (x - self.m) * (x - m),
            }
        }
    }

}

impl FoldState for Var {
    type Out = f32;

    fn into_result(self) -> Self::Out {
        if self.k > 1 { 
            self.s / self.k as f32
        } else {
            0.
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{ten, tensor::Axis, tf32};

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
    fn reduce_max() {
        assert_eq!(ten![1., 3., 2.].reduce_max(), ten![3.]);
    }

    #[test]
    fn reduce_min() {
        assert_eq!(ten!([1., 3., 2.]).reduce_min(), ten![1.]);
    }
    
    #[test]
    fn reduce_hypot() {
        assert_eq!(tf32!([3., 4.]).reduce_hypot(), tf32!([5.]));
        assert_eq!(tf32!([2., 2., 2., 2.]).reduce_hypot(), tf32!([4.]));
    }
    
    #[test]
    fn reduce_mean() {
        assert_eq!(tf32!([1.]).reduce_mean(), tf32!([1.]));
        assert_eq!(tf32!([1., 3.]).reduce_mean(), tf32!([2.]));
        // Axis::None behavior is to treat the tensor as flat
        assert_eq!(tf32!([[1., 3.], [4., 0.]]).reduce_mean(), tf32!([2.]));
    }

    #[test]
    fn reduce_mean_axis() {
        assert_eq!(tf32!([1.]).reduce_mean_axis(None), tf32!(1.));
        assert_eq!(tf32!([1., 3.]).reduce_mean_axis(None), tf32!(2.));
        assert_eq!(tf32!([[1., 3.], [4., 6.]]).reduce_mean_axis(None), tf32!([3.5]));
        assert_eq!(tf32!([[[1., 3.]], [[4., 6.]]]).reduce_mean_axis(None), tf32!([3.5]));

        assert_eq!(tf32!([1.]).reduce_mean_axis(Axis::axis(-1)), tf32!(1.));
        assert_eq!(tf32!([1., 3.]).reduce_mean_axis(-1), tf32!(2.));
        assert_eq!(tf32!([[1., 3.], [4., 6.]]).reduce_mean_axis(-1), tf32!([2., 5.]));
        assert_eq!(tf32!([[[1., 3.]], [[4., 6.]]]).reduce_mean_axis(-1), tf32!([[2.], [5.]]));

        assert_eq!(tf32!([1.]).reduce_mean_axis(0), tf32!(1.));
        assert_eq!(tf32!([1., 3.]).reduce_mean_axis(0), tf32!(2.));
        assert_eq!(tf32!([[1., 3.], [4., 6.]]).reduce_mean_axis(0), tf32!([2.5, 4.5]));
        assert_eq!(tf32!([[[1., 3.]], [[4., 6.]]]).reduce_mean_axis(0), tf32!([[2.5, 4.5]]));
    }

    #[test]
    fn reduce_std() {
        assert_eq!(tf32!([1.]).reduce_std(), tf32!(0.));
        assert_eq!(tf32!([1., 1.]).reduce_std(), tf32!(0.));
        assert_eq!(tf32!([2., 2., 2., 2.]).reduce_std(), tf32!(0.));

        assert_eq!(tf32!([1., 3., 2., 2.]).reduce_std(), tf32!(0.70710677));
        assert_eq!(tf32!([[1., 3.], [2., 2.]]).reduce_std(), tf32!(0.70710677));
        assert_eq!(tf32!([[1., 3.], [2., 2.]]).reduce_std_axis(None), tf32!(0.70710677));
        assert_eq!(tf32!([[1., 3.], [2., 2.]]).reduce_std_axis(-1), tf32!([1.0, 0.0]));
        assert_eq!(tf32!([[1., 3.], [2., 2.]]).reduce_std_axis(0), tf32!([0.5, 0.5]));

        assert_eq!(tf32!([1., 3.]).reduce_std(), tf32!(1.));
        assert_eq!(tf32!([1., 3., 3.]).reduce_std(), tf32!(0.94280905));
        assert_eq!(tf32!([1., 3., 1., 3.]).reduce_std(), tf32!(1.));
        assert_eq!(tf32!([1., 3., 4., 0.]).reduce_std(), tf32!(1.5811388));
        assert_eq!(tf32!([1., 3., 4., 0., 2.]).reduce_std(), tf32!(1.4142135));
    }

    #[test]
    fn reduce_var() {
        assert_eq!(tf32!([1.]).reduce_variance(), tf32!(0.));
        assert_eq!(tf32!([1., 1.]).reduce_variance(), tf32!(0.));
        assert_eq!(tf32!([2., 2., 2., 2.]).reduce_variance(), tf32!(0.));

        assert_eq!(tf32!([1., 3.]).reduce_variance(), tf32!(1.));
        assert_eq!(tf32!([1., 3., 1., 3.]).reduce_variance(), tf32!(1.));
        assert_eq!(tf32!([1., 3., 3.]).reduce_variance(), tf32!(0.8888889));
        assert_eq!(tf32!([1., 3., 4., 0.]).reduce_variance(), tf32!(2.5));
        assert_eq!(tf32!([1., 3., 4., 0., 2.]).reduce_variance(), tf32!(2.0));
    }
}
