use crate::Tensor;


pub fn diagflat(diag: impl Into<Tensor>) -> Tensor {
    let vec : Tensor = diag.into();

    assert!(vec.rank() == 1, "diagflat currently expects a 1d vector {:?}", vec.shape().as_slice());
    let n = vec.len();
    let size = n * n;

    let mut data = Vec::new();
    data.resize(size, 0.);

    for (i, value) in vec.iter().enumerate() {
        data[i * n + i] = *value;
    }

    Tensor::from_vec(data, [n, n])
}

impl Tensor {
    pub fn diagflat(&self) -> Tensor {
        diagflat(self)
    }
}

#[cfg(test)]
mod test {
    use crate::{init::diagflat, tf32};

    #[test]
    fn test_diagflat() {
        assert_eq!(
            diagflat(tf32!([1., 2., 3.])), tf32!([
            [1., 0., 0.],
            [0., 2., 0.],
            [0., 0., 3.],
        ]));
    }
}