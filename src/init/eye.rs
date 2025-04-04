use crate::tensor::Tensor;


pub fn eye(n: usize) -> Tensor {
    Tensor::init_indexed([n, n], |idx| {
        if idx[0] == idx[1] {
            1.
        } else {
            0.
        }
    })
}

pub fn identity(n: usize) -> Tensor {
    eye(n)
}

impl Tensor {
    pub fn eye(n: usize) -> Tensor {
        eye(n)
    }

    pub fn identity(n: usize) -> Tensor {
        identity(n)
    }
}

#[cfg(test)]
mod test {
    use eye::identity;

    use crate::{init::eye, tf32, tensor::Tensor};

    #[test]
    fn test_eye() {
        assert_eq!(eye(0), Tensor::empty().reshape([0, 0]));
        assert_eq!(eye(1), tf32!([[1.]]));
        assert_eq!(eye(2), tf32!([[1., 0.], [0., 1.]]));
        assert_eq!(eye(4), tf32!([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ]));
    }

    #[test]
    fn test_identity() {
        assert_eq!(identity(0), Tensor::empty().reshape([0, 0]));
        assert_eq!(identity(1), tf32!([[1.]]));
        assert_eq!(identity(2), tf32!([[1., 0.], [0., 1.]]));
        assert_eq!(identity(4), tf32!([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ]));
    }
}