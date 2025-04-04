use crate::tensor::{Dtype, Tensor, TensorData};

pub fn transpose<T: Clone + 'static>(tensor: impl Into<Tensor<T>>) -> Tensor<T> {
    let tensor: Tensor<T> = tensor.into();

    let cols = tensor.cols().max(1);
    let rows = tensor.rows().max(1);
    let size = tensor.shape().size();
    let n_inner = cols * rows;
    let batch = size / n_inner;

    let mut shape = tensor.shape().clone();
    if shape.rank() == 1 && cols > 1 {
        shape = shape.with_rank(2).with_col(rows).with_row(cols);
    } else {
        shape = shape.with_col(rows).with_row(cols);
    }

    unsafe {
        TensorData::<T>::unsafe_init(size, |o| {
            let x = tensor.as_slice();

            for n in 0..batch {
                for j in 0..rows {
                    for i in 0..cols {
                        o.add(n * n_inner + i * rows + j)
                            .write(x[n * n_inner + j * cols + i].clone());
                    }
                }
            }
        }).into_tensor(shape)
    }
}

impl<D: Dtype + Clone> Tensor<D> {
    #[inline]
    pub fn transpose(&self) -> Tensor<D> {
        transpose(self)
    }

    #[inline]
    pub fn t(&self) -> Tensor<D> {
        transpose(self)
    }
}

#[cfg(test)]
mod test {
    use crate::{prelude::*, array::transpose};
    
    #[test]
    fn test_transpose() {
        assert_eq!(
            transpose(&tf32!([1., 2.])),
            tf32!([[1.], [2.]]),
        );

        assert_eq!(
            transpose(&tf32!([[1., 2.], [3., 4.], [5., 6.]])),
            tf32!([[1., 3., 5.], [2., 4., 6.]]),
        );
    }
}
