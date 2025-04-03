use std::{any::type_name, cmp};

use crate::{
    linalg::blas::sgemm, tensor::{Tensor, TensorData}
};

use super::matmul::Transpose;

#[derive(Debug, Clone)]
struct Matvec;

#[derive(Debug, Clone)]
struct MatvecBackLeft;

#[derive(Debug, Clone)]
struct MatvecBackRight;

impl Tensor<f32> {
    pub fn matvec(&self, b: &Tensor<f32>) -> Tensor {
        matvec_t(&self, b, Transpose::None)
    }

    pub fn matvec_t(&self, b: &Tensor<f32>, transpose: impl TransposeMatvec) -> Tensor {
        matvec_t(&self, b, transpose)
    }
}

pub fn matvec(a: &Tensor<f32>, b: &Tensor<f32>) -> Tensor {
    // let node = NodeOp::new(&[a, b], Box::new(Matvec));

    let value = matvec_t_op(a, b, Transpose::None); //, node);

    // Tape::set_tensor(value)
    todo!();
}

pub fn matvec_t(
    a: &Tensor<f32>,
    x: &Tensor<f32>,
    _transpose: impl TransposeMatvec,
) -> Tensor<f32> {
    /*
    let node = NodeOp::new(&[a, x], Box::new(Matvec));

    let value = matvec_t_op(a, x, _transpose, node);

    Tape::set_tensor(value)
    */
    todo!();
}

fn matvec_t_op(
    a: &Tensor<f32>,
    x: &Tensor<f32>,
    transpose: impl TransposeMatvec,
) -> Tensor<f32> {
    assert!(a.rank() >= 2, "matrix[{}]-vector multiplication requires dim >= 2", a.rank());
    
    let batch : usize = a.broadcast_min(2, &x, cmp::min(2, x.rank()));

    let a_size = a.cols() * a.rows();

    let x_rows = cmp::max(1, x.rows());
    let x_size = x.cols() * x_rows;

    let (o_cols, o_rows) = transpose.o_size(a, x);
    let o_size = o_cols * o_rows;

    unsafe {
        TensorData::<f32>::unsafe_init(o_size * batch, |o| {
            for n in 0..batch {
                let a_ptr = a.as_wrap_ptr(n * a_size);
                let x_ptr = x.as_wrap_ptr(n * x_size);
                let o_ptr = o.add(n * o_size);

                transpose.sgemm(a, x, o_cols, o_rows, a_ptr, x_ptr, o_ptr);
            }
        }).into_tensor(x.shape().clone().with_col(o_cols))
    }
}

pub trait TransposeMatvec {
    fn o_size(&self, a: &Tensor<f32>, x: &Tensor<f32>) -> (usize, usize);

    fn sgemm(
        &self, 
        a: &Tensor<f32>, 
        x: &Tensor<f32>, 
        o_cols: usize,
        o_rows: usize,
        a_ptr: *const f32, 
        x_ptr: *const f32, 
        o_ptr: *mut f32,
    );
}

impl TransposeMatvec for Transpose {
    fn o_size(&self, a: &Tensor<f32>, x: &Tensor<f32>) -> (usize, usize) {
        match self {
            Transpose::None => {
                assert_eq!(a.cols(), x.cols(),
                    "matvec shapes do not match. A={:?} x={:?}",
                    &a.shape().as_vec(), &x.shape().as_vec());

                (a.rows(), cmp::max(1, x.rows()))
            },
            Transpose::TransposeA => {
                assert_eq!(a.rows(), x.cols(),
                    "matvec shapes do not match. A={:?} x={:?} for {:?}",
                    &a.shape().as_vec(), &x.shape().as_vec(), self);
                    
                (a.cols(), cmp::max(1, x.rows()))
            },
            Transpose::TransposeB => todo!(),
            Transpose::TransposeAB => todo!(),
        }
    }

    fn sgemm(
        &self, 
        a: &Tensor<f32>, 
        x: &Tensor<f32>,
        o_cols: usize,
        o_rows: usize,
        a_ptr: *const f32, 
        x_ptr: *const f32, 
        o_ptr: *mut f32,
    ) {
        match self {
            Transpose::None => {
                unsafe {
                    sgemm(
                        o_rows, x.cols(), o_cols,
                        1.,
                        x_ptr,
                        x.cols(), 1,
                        a_ptr,
                        1, a.cols(),
                        0.,
                        o_ptr,
                        o_cols, 1,
                    );
                }
            }
            Transpose::TransposeA => {
                unsafe {
                    sgemm(
                        o_rows, x.cols(), o_cols,
                        1.,
                        x_ptr,
                        x.cols(), 1,
                        a_ptr,
                        a.cols(), 1,
                        0.,
                        o_ptr,
                        o_cols, 1,
                    );
                }
            },
            Transpose::TransposeB => todo!(),
            Transpose::TransposeAB => todo!(),
        }
    }
}
/*
impl Operation<f32> for Matvec {
    fn name(&self) -> &str {
        type_name::<Self>()
    }
    
    fn f(
        &self,
        args: &[&Tensor],
        node: TensorId,
    ) -> Tensor {
        let value = matvec_t_op(args[0], args[1], Transpose::None, node);

        value
    }
}
*/

#[cfg(test)]
mod test {
    use crate::{ten, Tensor, linalg::matmul::Transpose, tf32};

    #[test]
    fn test_matvec_1_1() {
        let a = ten!([[2.]]);
        let b = ten!([3.]);

        assert_eq!(a.matvec(&b), ten!([6.]));
    }

    #[test]
    fn test_matvec_1_2() {
        let a = ten!([[1., 2.]]);
        let b = ten!([3., 4.]);

        assert_eq!(a.matvec(&b), ten!([11.]));
    }

    #[test]
    fn test_matvec_2_n() {
        let a = ten!([[1.], [2.]]);
        let b = ten!([2.]);
        assert_eq!(a.matvec(&b), ten!([2., 4.]));

        let a = ten!([[1., 2.], [2., 3.]]);
        let b = ten!([2., 3.]);
        assert_eq!(a.matvec(&b), ten!([8., 13.]));
    }

    #[test]
    fn test_matvec_3_n() {
        let a = ten!([[1.], [2.], [3.]]);
        let b = ten!([2.]);
        assert_eq!(a.matvec(&b), ten!([2., 4., 6.]));
    }

    #[test]
    fn test_matvec_t() {
        let a = ten!([[1., 4.], [2., 5.], [3., 6.]]);
        let b = ten!([10., 20.]);
        assert_eq!(a.matvec(&b), ten!([90., 120., 150.]));

        let a = ten!([[1., 2., 3.], [4., 5., 6.]]);
        let b = ten!([10., 20.]);
        assert_eq!(a.matvec_t(&b, Transpose::TransposeA), ten!([90., 120., 150.]));
    }

    #[test]
    #[should_panic]
    fn matvec_2x1_by_2() {
        // assert_eq!(a.matvec(&b), tensor!([2., 20.]));
        
        let a = tf32!([[10.], [20.]]);
        let x = tf32!([1., 3.]);

        assert_eq!(a.matvec(&x), tf32!([[10.], [20.]]));
    }
    
}
