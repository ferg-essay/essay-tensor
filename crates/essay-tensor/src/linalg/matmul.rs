use std::sync::Arc;

use crate::{tensor::{Tensor, TensorUninit}, module::{EvalOp}};

#[derive(Clone, Debug)]
pub enum Transpose {
    None,
    TransposeA,
    TransposeB,
    TransposeAB,
}

pub trait TransposeMatmul {
    fn inner_len(
        &self, 
        a_dim: [usize; 2], 
        b_dim: [usize; 2], 
    ) -> usize;

    fn o_stride(&self, a_dim: [usize; 2], b_dim: [usize; 2]) -> [usize; 2];

    fn a_stride(&self, a_dim: [usize; 2]) -> [usize; 2];

    fn b_stride(&self, b_dim: [usize; 2]) -> [usize; 2];
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
    assert_eq!(&a.shape()[2..], &b.shape()[2..], "matmul batch shape must match");

    let a_dims = [a.shape()[0], a.shape()[1]];
    let b_dims = [b.shape()[0], b.shape()[1]];

    let o_stride = transpose.o_stride(a_dims, b_dims);
    let inner_len = transpose.inner_len(a_dims, b_dims);

    let a_stride = transpose.a_stride(a_dims);
    let b_stride = transpose.b_stride(b_dims);

    let batch_len : usize = a.shape()[2..].iter().product();
    let a_size = a_dims[0] * a_dims[1];
    let b_size = b_dims[0] * b_dims[1];
    let o_size = o_stride[0] * o_stride[1];

    unsafe {
        let out = TensorUninit::<f32>::new(o_size * batch_len);

        for batch in 0..batch_len {
            let a_ptr = a.data().as_ptr().add(a_size * batch);
            let b_ptr = b.data().as_ptr().add(b_size * batch);
            let out_ptr = out.as_ptr().add(o_size * batch);
        
            naive_matmul(
                out_ptr,
                o_stride,
                inner_len,
                a_ptr,
                a_stride,
                b_ptr,
                b_stride,
            );
        }

        let mut o_shape = b.shape().clone();
        o_shape[0] = o_stride[0];
        o_shape[1] = o_stride[1];

        //a.next_binop(&b, out.init(), o_shape, Matmul)
        Tensor::new(out.init(), &o_shape)
    }
}

unsafe fn naive_matmul(
    o_ptr: *mut f32,
    o_stride: [usize; 2],
    inner_len: usize,
    a_ptr: *const f32,
    a_stride: [usize; 2],
    b_ptr: *const f32,
    b_stride: [usize; 2],
) {
    for row in 0..o_stride[1] {
        for col in 0..o_stride[0] {
            let mut v: f32 = 0.;
            for k in 0..inner_len {
                let a = *a_ptr.add(row * a_stride[1] + k * a_stride[0]);
                let b = *b_ptr.add(col * b_stride[1] + k * b_stride[1]);
                v += a * b;
            }

            *o_ptr.add(row * o_stride[1] + col) = v;
        }
    }
}

impl TransposeMatmul for Transpose {
    #[inline]
    fn inner_len(
        &self, 
        a: [usize; 2], 
        b: [usize; 2],
    ) -> usize {
        match self {
            Transpose::None => {
                assert_eq!(a[0], b[1], "left columns must match right rows");
                a[0]
            },
            Transpose::TransposeA => {
                assert_eq!(a[1], b[1], "left rows must match right rows {:?}", &self);
                a[1]
            },
            Transpose::TransposeB => {
                assert_eq!(a[0], b[0], "left columns must match right columns {:?}", &self);
                a[0]
            },
            Transpose::TransposeAB => {
                assert_eq!(a[1], b[0], "left rows must match right columns {:?}", &self);
                a[1]
            },
        }
    }

    #[inline]
    fn o_stride(&self, a: [usize; 2], b: [usize; 2]) -> [usize; 2] {
        match self {
            Transpose::None => [b[0], a[1]],
            Transpose::TransposeA => [b[0], a[0]],
            Transpose::TransposeB => [b[1], a[1]],
            Transpose::TransposeAB => [b[1], a[1]],
        }
    }

    fn a_stride(&self, a: [usize; 2]) -> [usize; 2] {
        match self {
            Transpose::None => [1, a[0]],
            Transpose::TransposeA => [a[0], 1],
            Transpose::TransposeB => [1, a[0]],
            Transpose::TransposeAB => [1, a[0]],
        }
    }

    fn b_stride(&self, b: [usize; 2]) -> [usize; 2]
    {
        match self {
            Transpose::None => [b[0], 1],
            Transpose::TransposeA => [b[0], 1],
            Transpose::TransposeB => [1, b[0]],
            Transpose::TransposeAB => [1, b[0]],
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
