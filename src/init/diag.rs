use crate::tensor::Tensor;


pub fn diagflat(diag: impl Into<Tensor>) -> Tensor {
    let diag : Tensor = diag.into();

    assert!(diag.rank() == 1, "diagflat currently expects a 1d vector {:?}", diag.shape().as_vec());
    let n = diag.len();

    Tensor::init_indexed([n, n], |idx| {
        if idx[0] == idx[1] {
            diag[idx[0]]
        } else {
            0.
        }
    })
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