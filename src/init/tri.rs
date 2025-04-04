use crate::tensor::{TensorData, Tensor};


pub fn tri(n: usize) -> Tensor {
    let size = n * n;

    unsafe {
        TensorData::<f32>::unsafe_init(size, |o| {
            for j in 0..n {
                for i in 0..n {
                    o.add(j * n + i)
                        .write(if i <= j { 1. } else { 0. });
                }
            }
        }).into_tensor([n, n])
    }
}

impl Tensor {
    pub fn tri(n: usize) -> Tensor {
        tri(n)
    }
}


impl Tensor {
    pub fn tril(tensor: impl Into<Tensor>) -> Tensor {
        tril(tensor)
    }
}

pub fn tril(tensor: impl Into<Tensor>) -> Tensor {
    let tensor = tensor.into();

    assert!(tensor.rank() >= 2);

    let size = tensor.shape().size();

    unsafe {
        TensorData::<f32>::unsafe_init(size, |o| {
            let x = tensor.as_slice();

            let rows = tensor.rows();
            let cols = tensor.cols();
            let n = size / (rows * cols);

            for k in 0..n {
                for j in 0..rows {
                    for i in 0..cols {
                        let index = k * rows * cols + j * cols + i;

                        o.add(index)
                            .write(if i <= j { x[index] } else { 0. });
                    }
                }
            }
        }).into_tensor(tensor.shape())
    }
}

impl Tensor {
    pub fn triu(tensor: impl Into<Tensor>) -> Tensor {
        triu(tensor)
    }
}

pub fn triu(tensor: impl Into<Tensor>) -> Tensor {
    let tensor = tensor.into();

    assert!(tensor.rank() >= 2);

    let size = tensor.shape().size();

    unsafe {
        TensorData::<f32>::unsafe_init(size, |o| {
            let x = tensor.as_slice();

            let rows = tensor.rows();
            let cols = tensor.cols();
            let n = size / (rows * cols);

            for k in 0..n {
                for j in 0..rows {
                    for i in 0..cols {
                        let index = k * rows * cols + j * cols + i;

                        o.add(index)
                            .write(if j <= i { x[index] } else { 0. });
                    }
                }
            }
        }).into_tensor(tensor.shape())
    }
}
#[cfg(test)]
mod test {
    use tri::{tril, triu};

    use crate::{init::{tri}, tf32};

    #[test]
    fn test_tri() {
        assert_eq!(tri(1), tf32!([[1.]]));
        assert_eq!(tri(2), tf32!([[1., 0.], [1., 1.]]));
        assert_eq!(tri(4), tf32!([
            [1., 0., 0., 0.],
            [1., 1., 0., 0.],
            [1., 1., 1., 0.],
            [1., 1., 1., 1.],
        ]));
    }

    #[test]
    fn test_tril() {
        assert_eq!(tril(tf32!([
            [1., 2., 3., 4.],
            [11., 12., 13., 14.],
            [21., 22., 23., 24.],
            [31., 32., 33., 34.],
        ])), tf32!([
            [ 1.,  0.,  0.,  0.],
            [11., 12.,  0.,  0.],
            [21., 22., 23.,  0.],
            [31., 32., 33., 34.],
        ]));
    }

    #[test]
    fn test_triu() {
        assert_eq!(triu(tf32!([
            [1., 2., 3., 4.],
            [11., 12., 13., 14.],
            [21., 22., 23., 24.],
            [31., 32., 33., 34.],
        ])), tf32!([
            [1., 2., 3., 4.],
            [0., 12., 13., 14.],
            [0., 0., 23., 24.],
            [0., 0., 0., 34.],
        ]));
    }
}