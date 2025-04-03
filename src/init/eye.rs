use crate::{tensor::TensorData, Tensor};


pub fn eye(n: usize) -> Tensor {
    let size = n * n;

    unsafe {
        TensorData::<f32>::unsafe_init(size, |o| {
            for i in 0..n {
                for j in 0..n {
                    o.add(i * n + j).write(0.);
                }
            }

            for i in 0..n {
                o.add(i * n + i).write(1.);
            }
        }).into_tensor([n, n])
    }
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

    use crate::{init::eye, tf32, Tensor};

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