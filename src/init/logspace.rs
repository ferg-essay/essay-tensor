use std::cmp;

use crate::tensor::{Tensor, TensorData};

pub fn logspace(
    start: impl Into<Tensor>, 
    end: impl Into<Tensor>, 
    len: usize,
) -> Tensor {
    logspace_opt(start, end, len, None)
}

pub fn logspace_opt(
    start: impl Into<Tensor>, 
    end: impl Into<Tensor>, 
    len: usize,
    opt: impl Into<Logspace>
) -> Tensor {
    let start = start.into();
    let end = end.into();

    assert_eq!(start.shape(), end.shape(), 
        "logspace shapes must agree start={:?} end={:?}",
        start.shape(), end.shape());

    let opt : Logspace = opt.into();
    let base = match opt.base {
        Some(base) => base,
        None => 10.
    };

    let batch = cmp::max(1, start.size());
    let size = batch * len;

    let o_shape = start.shape().clone().with_cols(len);

    unsafe {
        TensorData::<f32>::unsafe_init(size, |o| {
            for n in 0..batch {
                let start = start[n];
                let end = end[n];

                assert!(start <= end);

                let step = if len > 1 {
                    (end - start) / (len - 1) as f32
                } else {
                    0.
                };

                for k in 0..len {
                    let v = start + step * k as f32;
                    o.add(k * batch + n).write(base.powf(v));
                }
            }
        }).into_tensor(o_shape)
    }
}

impl Tensor {
    pub fn logspace(&self, end: &Tensor, len: usize) -> Tensor {
        logspace(self, end, len)
    }
}

#[derive(Debug, Default)]
pub struct Logspace {
    base: Option<f32>,
}

impl Logspace {
    pub fn base(base: f32) -> Self {
        Self {
            base: Some(base)
        }
    }
}

impl From<Option<f32>> for Logspace {
    fn from(base: Option<f32>) -> Self {
        Self {
            base,
        }
    }
}

impl From<f32> for Logspace {
    fn from(base: f32) -> Self {
        Logspace { base: Some(base) }
    }
}

#[cfg(test)]
mod test {
    use crate::init::logspace_opt;
    use crate::ten;
    use crate::init::logspace::logspace;

    #[test]
    fn logspace_0_2_3() {
        assert_eq!(logspace(0., 2., 3), ten![1., 10., 100.]);
    }

    #[test]
    fn logspace_opt_0_2_3() {
        assert_eq!(logspace_opt(0., 2., 3, Some(2.)), ten![1., 2., 4.]);
    }
}