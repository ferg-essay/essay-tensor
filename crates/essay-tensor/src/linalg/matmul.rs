use crate::{tensor::{Tensor, TensorUninit}, function::{EvalOp}, linalg::blas::sgemm};

#[derive(Clone, Debug)]
pub enum Transpose {
    None,
    TransposeA,
    TransposeB,
    TransposeAB,
}

pub trait TransposeMatmul {
    fn mkn(
        &self, 
        a: &Tensor,
        b: &Tensor,
    ) -> (usize, usize, usize);

    unsafe fn sgemm(
        &self, 
        a: &Tensor, b: &Tensor,
        a_ptr: *const f32,
        b_ptr: *const f32,
        o_ptr: *mut f32,
    );
}

#[derive(Debug, Clone)]
struct Matmul;

impl Tensor {
    pub fn matmul(&self, b: &Tensor) -> Tensor {
        matmul_t(self, b, Transpose::None)
    }

    pub fn matmul_t(&self, b: &Tensor, transpose: Transpose) -> Tensor {
        matmul_t(self, b, transpose)
    }
}

pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    matmul_t(a, b, Transpose::None)
}

pub fn matmul_t<T: TransposeMatmul>(a: &Tensor, b: &Tensor, transpose: T) -> Tensor {
    //assert_eq!(M, N, "matrix multiplication requires matching dim >= 2");
    assert!(a.rank() > 1, "matrix multiplication requires rank >= 2");
    assert_eq!(&a.shape().as_subslice(2..), &b.shape().as_subslice(2..), "matmul batch shape must match");

    let (m, _, n) = transpose.mkn(a, b);

    let batch_len : usize = a.shape().sublen(0..a.rank() - 2);
    let a_size = a.rows() * a.cols();
    let b_size = b.rows() * b.cols();
    let o_size = m * n;

    unsafe {
        let out = TensorUninit::<f32>::new(o_size * batch_len);

        for batch in 0..batch_len {
            let a_ptr = a.as_ptr().add(a_size * batch);
            let b_ptr = b.as_ptr().add(b_size * batch);
            let c_ptr = out.as_ptr().add(o_size * batch);
        
            transpose.sgemm(a, b, a_ptr, b_ptr, c_ptr);
        }

        let mut o_shape = Vec::from(b.shape().as_slice());
        let len = o_shape.len();
        o_shape[len - 1] = m;
        o_shape[len - 2] = n;

        Tensor::from_uninit(out, o_shape)
    }
}

impl TransposeMatmul for Transpose {
    #[inline]
    fn mkn(
        &self, 
        a: &Tensor,
        b: &Tensor,
    ) -> (usize, usize, usize) {
        match self {
            Transpose::None => {
                assert_eq!(a.cols(), b.rows(), "matmul shape does not match. A={:?} B={:?}",
                    a.shape().as_slice(), b.shape().as_slice());

                (a.rows(), a.cols(), b.cols())
            },
            Transpose::TransposeA => {
                assert_eq!(a.rows(), b.rows(), "matmul shape does not match. A={:?} B={:?} for {:?}", 
                    a.shape().as_slice(), b.shape().as_slice(), &self);

                (a.cols(), a.rows(), b.cols())
            },
            Transpose::TransposeB => {
                assert_eq!(a.cols(), b.cols(), "matmul shape does not match. A={:?} B={:?} for {:?}", 
                    a.shape().as_slice(), b.shape().as_slice(), &self);

                (a.rows(), a.rows(), b.rows())
            },
            Transpose::TransposeAB => {
                assert_eq!(a.rows(), b.cols(), "matmul shape does not match. A={:?} B={:?} for {:?}", 
                    a.shape().as_slice(), b.shape().as_slice(), &self);

                (a.cols(), a.rows(), b.rows())
            },
        }
    }

    #[inline]
    unsafe fn sgemm(
        &self, 
        a: &Tensor,
        b: &Tensor,
        a_ptr: *const f32,
        b_ptr: *const f32,
        o_ptr: *mut f32,
    ) {
        match self {
            Transpose::None => {
                sgemm(
                    a.rows(), a.cols(), b.cols(),
                    1.,
                    a_ptr, a.cols(), 1,
                    b_ptr, b.cols(), 1,
                    0.,
                    o_ptr, b.cols(), 1,
                );
            }
            Transpose::TransposeA => {
                sgemm(
                    a.cols(), a.rows(), b.cols(),
                    1.,
                    a_ptr, 1, a.cols(),
                    b_ptr, b.cols(), 1,
                    0.,
                    o_ptr, b.cols(), 1,
                );
            }
            Transpose::TransposeB => {
                sgemm(
                    a.rows(), a.cols(), b.rows(),
                    1.,
                    a_ptr, a.cols(), 1,
                    b_ptr, 1, b.cols(),
                    0.,
                    o_ptr, b.rows(), 1,
                );
            }
            Transpose::TransposeAB => {
                sgemm(
                    a.cols(), a.rows(), b.rows(),
                    1.,
                    a_ptr, 1, a.cols(),
                    b_ptr, 1, b.cols(),
                    0.,
                    o_ptr, b.rows(), 1,
                );
            }
        }
    }
}

impl EvalOp for Matmul {
    fn eval(
        &self,
        _args: &[&Tensor],
    ) -> Tensor {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use crate::{tensor, Tensor, linalg::matmul::Transpose};

    #[test]
    fn test_matmul_1() {
        let a = tensor!([[2.]]);
        let b = tensor!([[3.]]);

        assert_eq!(a.matmul(&b), tensor!([[6.]]));
    }

    #[test]
    fn test_matmul_vectors() {
        let a = tensor!([[1., 2.]]);
        let b = tensor!([[1.], [3.]]);
        assert_eq!(a.matmul(&b), tensor!([[7.]]));

        let a = tensor!([[1.], [3.]]);
        let b = tensor!([[1., 2.]]);
        assert_eq!(a.matmul(&b), tensor!([[1., 2.], [3., 6.]]));
    }

    #[test]
    fn test_matmul_square() {
        let id = tensor!([[1., 0.], [0., 1.]]);
        assert_eq!(&id.clone().matmul(&id), &id);

        let a = tensor!([[1., 0.], [0., 2.]]);
        assert_eq!(&id.clone().matmul(&a), &a);
        assert_eq!(&a.clone().matmul(&id), &a);
        assert_eq!(&a.clone().matmul(&a),
            &tensor!([[1., 0.], [0., 4.]]));

        let top = tensor!([[1., 1.], [0., 1.]]);
        assert_eq!(top.matmul(&top), tensor!([[1., 2.], [0., 1.]]));

        let bot = tensor!([[1., 0.], [1., 1.]]);
        assert_eq!(bot.matmul(&bot), tensor!([[1., 0.], [2., 1.]]));
    }

    #[test]
    fn test_matmul_2x3() {
        let a = tensor!([[1., 0., 2.], [0., 1., 10.]]);
        let b = tensor!([[1., 0.], [0., 1.], [3., 4.]]);
        assert_eq!(a.matmul(&b), tensor!([[7., 8.], [30., 41.]]));
        assert_eq!(b.matmul(&a), tensor!([
            [1., 0., 2.],
            [0., 1., 10.],
            [3., 4., 46.]]));
    }

    #[test]
    fn matmul_transpose_none() {
        let a = tensor!([[1., 2.]]);
        let b = tensor!([[10.], [20.]]);

        assert_eq!(a.matmul_t(&b, Transpose::None), tensor!([[50.]]));
    }

    #[test]
    fn matmul_transpose_a() {
        let a = tensor!([[1.], [2.]]);
        let b = tensor!([[10.], [20.]]);

        assert_eq!(a.matmul_t(&b, Transpose::TransposeA), tensor!([[50.]]));
    }

    #[test]
    fn matmul_transpose_b() {
        let a = tensor!([[1., 2.]]);
        let b = tensor!([[10., 20.]]);

        assert_eq!(a.matmul_t(&b, Transpose::TransposeB), tensor!([[50.]]));
    }

    #[test]
    fn matmul_transpose_ab() {
        let a = tensor!([[1.], [2.]]);
        let b = tensor!([[10., 20.]]);

        assert_eq!(a.matmul_t(&b, Transpose::TransposeAB), tensor!([[50.]]));
    }

    #[test]
    fn matmul_2_2_transpose() {
        let a = tensor!([[1., 2.], [3., 4.]]);
        let b = tensor!([[10., 20.], [30., 40.]]);

        assert_eq!(a.matmul_t(&b, Transpose::None), 
            tensor!([[70., 100.], [150., 220.]]));

        assert_eq!(a.matmul_t(&b, Transpose::TransposeA), 
            tensor!([[100., 140.], [140., 200.]]));

        assert_eq!(a.matmul_t(&b, Transpose::TransposeB), 
            tensor!([[50., 110.], [110., 250.]]));
    }

    #[test]
    fn matmul_1_2_2_3_transpose() {
        let a = tensor!([[1., 2.]]);
        let b = tensor!([[10., 20., 30.], [40., 50., 60.]]);

        assert_eq!(a.matmul_t(&b, Transpose::None), 
            tensor!([[90., 120., 150.]]));

        let a = tensor!([[1.], [2.]]);
        let b = tensor!([[10., 20., 30.], [40., 50., 60.]]);

        assert_eq!(a.matmul_t(&b, Transpose::TransposeA), 
            tensor!([[90., 120., 150.]]));

        let a = tensor!([[1., 2.]]);
        let b = tensor!([[10., 40.], [20., 50.], [30., 60.]]);
    
        assert_eq!(a.matmul_t(&b, Transpose::TransposeB), 
            tensor!([[90., 120., 150.]]));
    }
}
