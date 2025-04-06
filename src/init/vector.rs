//
// Vector-like initializers
//

use std::cmp;

use crate::tensor::{Shape, Tensor, unsafe_init};

pub fn arange(start: f32, end: f32, step: f32) -> Tensor {
    assert!(step != 0.);

    let mut start = start;

    if start <= end && step > 0. {
        let mut vec = Vec::<f32>::new();

        while start < end {
            vec.push(start);
            start += step;
        }

        Tensor::from(vec)
    } else if end <= start && step < 0. {
        let mut vec = Vec::<f32>::new();

        while end < start {
            vec.push(start);
            start -= step;
        }

        Tensor::from(vec)
    } else {
        panic!("invalid arguments start={} end={} step={}", start, end, step);
    }
}

pub fn geomspace(
    start: impl Into<Tensor>, 
    end: impl Into<Tensor>, 
    len: usize,
) -> Tensor {
    let start = start.into();
    let end = end.into();

    assert_eq!(start.shape(), end.shape(), 
        "geomspace shapes must agree start={:?} end={:?}",
        start.shape(), end.shape());

    assert_eq!(start.shape(), end.shape());

    let batch = cmp::max(1, start.size());
    let size = batch * len;

    let o_shape = if start.size() == 1 {
        Shape::from([len])
    } else {
        start.shape().clone().insert(0, len)
    };

    unsafe {
        unsafe_init::<f32>(size, o_shape, |o| {
            for n in 0..batch {
                let start = start[n].ln();
                let end = end[n].ln();

                assert!(start <= end);

                let step = if len > 1 {
                    (end - start) / (len - 1) as f32
                } else {
                    0.
                };

                for k in 0..len {
                    let v = start + step * k as f32;
                    o.add(k * batch + n).write(v.exp());
                }
            }
        })
    }
}

pub fn linspace(start: impl Into<Tensor>, end: impl Into<Tensor>, len: usize) -> Tensor {
    let start = start.into();
    let end = end.into();

    assert_eq!(start.shape(), end.shape(), 
        "linspace shapes must agree start={:?} end={:?}",
        start.shape(), end.shape());

    let batch = cmp::max(1, start.size());
    let size = batch * len;

    let o_shape = if start.size() == 1 {
        Shape::from([len])
    } else {
        start.shape().clone().insert(0, len)
    };

    unsafe {
        unsafe_init::<f32>(size, o_shape, |o| {
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
                    o.add(k * batch + n).write(v);
                }
            }
        })
    }
}

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
        unsafe_init::<f32>(size, o_shape, |o| {
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
        })
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

pub fn one_hot(indices: impl AsRef<[usize]>, depth: usize) -> Tensor {
    let mut vec = Vec::<f32>::new();
    vec.resize(depth, 0.);

    for i in indices.as_ref() {
        vec[*i] = 1.;
    }

    Tensor::from(vec)
}

impl Tensor {
    pub fn arange(start: f32, end: f32, step: f32) -> Tensor {
        arange(start, end, step)
    }

    pub fn linspace(&self, end: &Tensor, len: usize) -> Tensor {
        linspace(self, end, len)
    }

    pub fn geomspace(&self, end: &Tensor, len: usize) -> Tensor {
        geomspace(self, end, len)
    }

    pub fn one_hot(indices: impl AsRef<[usize]>, depth: usize) -> Tensor {
        one_hot(indices, depth)
    }
}

#[cfg(test)]
mod test {
    use crate::init::{geomspace, logspace, logspace_opt};
    use crate::init::vector::one_hot;
    use crate::tensor::Tensor;
    use crate::{init::linspace, ten};

    #[test]
    fn linspace_0_4_5() {
        assert_eq!(linspace(0., 1., 2), ten![0., 1.]);
        assert_eq!(linspace(0., 4., 5), ten![0., 1., 2., 3., 4.]);

        assert_eq!(linspace(0., 4., 0), Tensor::<f32>::from(Vec::<f32>::new()));
    }

    #[test]
    fn linspace_vector_n() {
        assert_eq!(linspace([0.], [1.], 2), ten![[0.], [1.]]);
        assert_eq!(
            linspace([0., 0., 0.], [1., 2., 3.], 2), 
            ten![[0., 0., 0.], [1., 2., 3.]]
        );
    }

    #[test]
    fn linspace_tensor() {
        assert_eq!(linspace([[0.]], [[1.]], 2), ten![[[0.]], [[1.]]]);
        assert_eq!(
            linspace([[0., 0., 0.], [0., 0., 0.]], [[1., 2., 3.], [10., 20., 30.]], 3), 
            ten![
                [[0.0, 0.0, 0.0], [0., 0., 0.,]],
                [[0.5, 1.0, 1.5], [5., 10., 15.,]],
                [[1.0, 2.0, 3.0], [10., 20., 30.]]
            ]
        );
    }

    #[test]
    fn logspace_0_2_3() {
        assert_eq!(logspace(0., 2., 3), ten![1., 10., 100.]);
    }

    #[test]
    fn logspace_opt_0_2_3() {
        assert_eq!(logspace_opt(0., 2., 3, Some(2.)), ten![1., 2., 4.]);
    }

    #[test]
    fn geomspace_1_4_3() {
        assert_eq!(geomspace(1., 4., 3), ten![1., 2., 4.]);
        assert_eq!(geomspace(1., 8., 4), ten![1., 2., 4., 8.]);
        assert_eq!(
            geomspace(1., 8., 7),
            ten![1., 1.4142135, 2., 2.828427, 4., 5.656854, 8.]
        );
    }

    #[test]
    fn basic_one_hot() {
        assert_eq!(one_hot([0], 3), ten![1., 0., 0.]);
        assert_eq!(one_hot([1], 3), ten![0., 1., 0.]);
        assert_eq!(one_hot(vec![2], 3), ten![0., 0., 1.]);
        assert_eq!(one_hot(vec![0, 2], 3), ten![1., 0., 1.]);
    }
}