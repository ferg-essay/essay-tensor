use crate::tensor::{Axis, Tensor};

pub fn reduce_std(tensor: &Tensor) -> Tensor {
    tensor.fold_into(Acc::default(), |s, v| s.update(*v))
}

pub fn reduce_std_axis(tensor: &Tensor, axis: impl Into<Axis>) -> Tensor {
    tensor.fold_axis_into(axis, Acc::default(), |s, v| s.update(*v))
}

#[derive(Clone)]
struct Acc {
    k: usize,
    m: f32,
    s: f32,
}

impl Acc {
    fn update(self, x: f32) -> Self {
        // from Welford 1962
        if self.k == 0 {
            Acc {
                k: 1,
                m: x,
                s: 0.,
            }
        } else {
            let k = self.k + 1;
            let m = self.m + (x - self.m) / k as f32;

            Acc {
                k,
                m, 
                s: self.s + (x - self.m) * (x - m),
            }
        }
    }
}

impl Default for Acc {
    fn default() -> Self {
        Self { 
            k: 0,
            s: 0.,
            m: 0.,
        }
    }
}

impl From<Acc> for f32 {
    fn from(value: Acc) -> Self {
        if value.k > 1 { 
            (value.s / value.k as f32).sqrt()
        } else {
            0.
        }
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn reduce_std() {
        assert_eq!(tf32!([1.]).reduce_std(), tf32!(0.));
        assert_eq!(tf32!([1., 1.]).reduce_std(), tf32!(0.));
        assert_eq!(tf32!([2., 2., 2., 2.]).reduce_std(), tf32!(0.));

        assert_eq!(tf32!([1., 3.]).reduce_std(), tf32!(1.));
        assert_eq!(tf32!([1., 3., 1., 3.]).reduce_std(), tf32!(1.));
        assert_eq!(tf32!([1., 3., 3.]).reduce_std(), tf32!(0.94280905));
        assert_eq!(tf32!([1., 3., 4., 0.]).reduce_std(), tf32!(1.5811388));
        assert_eq!(tf32!([1., 3., 4., 0., 2.]).reduce_std(), tf32!(1.4142135));
    }
}