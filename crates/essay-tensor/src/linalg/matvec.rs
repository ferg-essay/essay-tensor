use std::{rc::Rc, ops};

use crate::tensor::{Tensor, TensorData, Dtype};

use super::matmul;

enum Transpose {
    None,
    TransposeA,
    TransposeB,
}

pub fn matvec<const N:usize, const M:usize>(
    a: &Tensor<N>, 
    b: &Tensor<M>
) -> Tensor<M> {
    assert!(N > 1, "matrix multiplication requires dim >= 2");
    assert_eq!(N, M + 1);
    assert_eq!(a.shape()[2..], b.shape()[1..], "matmul batch shape must match");
    assert_eq!(a.shape()[0], b.shape()[0], "matmul a.shape[0] must equal b.shape[1]");

    let n : usize = a.shape()[2..].iter().product();

    let cols = a.shape()[0];
    let rows = a.shape()[1];

    let a_size = a.shape()[0] * a.shape()[1];
    let b_size = b.shape()[0];
    let o_size = a.shape()[1];

    unsafe {
        let mut out = TensorData::<f32>::new_uninit(o_size * n);

        let mut a_start = 0;
        let mut b_start = 0;
        let mut o_start = 0;
    
        for _ in 0..n {
            naive_matvec_f32(
                &mut out, 
                o_start,
                a,
                a_start,
                b,
                b_start,
                cols,
                rows,
            );

            a_start += a_size;
            b_start += b_size;
            o_start += o_size;
        }

        let mut o_shape = b.shape().clone();
        o_shape[0] = a.shape()[1];
    
        Tensor::new(Rc::new(out), o_shape)
    }
}

impl<const N:usize> Tensor<N> {
    pub fn matvec<const M:usize>(&self, b: &Tensor<M>) -> Tensor<M> {
        matvec(&self, b)
    }
}

unsafe fn naive_matvec_f32<const N:usize, const M:usize>(
    out: &mut TensorData<f32>, 
    out_start: usize,
    a: &Tensor<N, f32>, 
    a_start: usize,
    b: &Tensor<M, f32>,
    b_start: usize,
    cols: usize,
    rows: usize,
) {
    let a_stride = a.shape()[0];

    let a_ptr = a.buffer().ptr();
    let b_ptr = b.buffer().ptr();
    let out_ptr = out.ptr();

    let mut a_row = a_start;

    for row in 0..rows {
        let mut v: f32 = 0.;
        for k in 0..a_stride {
            v += a_ptr.add(a_row + k).read() * b_ptr.add(b_start + k).read();
        }

        out_ptr.add(out_start + row).write(v);
        a_row += a_stride;
    }
}

#[cfg(test)]
mod test {
    use crate::{tensor, Tensor};

    use super::matvec;

    #[test]
    fn test_matvec_1_1() {
        let a = tensor!([[2.]]);
        let b = tensor!([3.]);

        assert_eq!(matvec(&a, &b), tensor!([6.]));
    }

    #[test]
    fn test_matvec_1_2() {
        let a = tensor!([[1., 2.]]);
        let b = tensor!([3., 4.]);

        assert_eq!(matvec(&a, &b), tensor!([11.]));
    }

    #[test]
    fn test_matvec_2_n() {
        let a = tensor!([[1.], [2.]]);
        let b = tensor!([2.]);
        assert_eq!(matvec(&a, &b), tensor!([2., 4.]));

        let a = tensor!([[1., 2.], [2., 3.]]);
        let b = tensor!([2., 3.]);
        assert_eq!(matvec(&a, &b), tensor!([8., 13.]));
    }

    #[test]
    fn test_matvec_3_n() {
        let a = tensor!([[1.], [2.], [3.]]);
        let b = tensor!([2.]);
        assert_eq!(matvec(&a, &b), tensor!([2., 4., 6.]));
    }
}