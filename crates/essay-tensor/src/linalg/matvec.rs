use crate::tensor::{Tensor, Op, BoxOp, TensorUninit};

enum Transpose {
    None,
    TransposeA,
    TransposeB,
}

#[derive(Debug, Clone)]
struct Matvec;

impl Op for Matvec {
    fn box_clone(&self) -> BoxOp {
        Box::new(Matvec)
    }
}

impl<const N:usize> Tensor<N, f32> {
    pub fn matvec<const M:usize>(
        self,
        b: Tensor<M, f32>
    ) -> Tensor<M, f32> {
        assert!(N > 1, "matrix multiplication requires dim >= 2");
        assert_eq!(N, M + 1);
        assert_eq!(self.shape()[2..], b.shape()[1..], "matmul batch shape must match");
        assert_eq!(self.shape()[0], b.shape()[0], "matmul a.shape[0] must equal b.shape[1]");

        let n : usize = self.shape()[2..].iter().product();

        let cols = self.shape()[0];
        let rows = self.shape()[1];

        let a_size = self.shape()[0] * self.shape()[1];
        let b_size = b.shape()[0];
        let o_size = self.shape()[1];

        unsafe {
            let mut out = TensorUninit::<f32>::new(o_size * n);

            let mut a_start = 0;
            let mut b_start = 0;
            let mut o_start = 0;
    
            for _ in 0..n {
                naive_matvec_f32(
                    &mut out, 
                    o_start,
                    &self,
                    a_start,
                    &b,
                    b_start,
                    cols,
                    rows,
                );

                a_start += a_size;
                b_start += b_size;
                o_start += o_size;
            }

            let mut o_shape = b.shape().clone();
            o_shape[0] = self.shape()[1];
            // Tensor::new(Rc::new(out), o_shape)
            self.next_binop(&b, out.init(), o_shape, Matvec.box_clone())
        }
    }
}

unsafe fn naive_matvec_f32<const N:usize, const M:usize>(
    out: &mut TensorUninit<f32>, 
    out_start: usize,
    a: &Tensor<N, f32>, 
    a_start: usize,
    b: &Tensor<M, f32>,
    b_start: usize,
    cols: usize,
    rows: usize,
) {
    let a_stride = a.shape()[0];

    let a_ptr = a.buffer().as_ptr();
    let b_ptr = b.buffer().as_ptr();
    let out_ptr = out.as_ptr();

    let mut a_row = a_start;

    for row in 0..rows {
        let mut v = 0.0;
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

    #[test]
    fn test_matvec_1_1() {
        let a = tensor!([[2.]]);
        let b = tensor!([3.]);

        assert_eq!(a.matvec(b), tensor!([6.]));
    }

    #[test]
    fn test_matvec_1_2() {
        let a = tensor!([[1., 2.]]);
        let b = tensor!([3., 4.]);

        assert_eq!(a.matvec(b), tensor!([11.]));
    }

    #[test]
    fn test_matvec_2_n() {
        let a = tensor!([[1.], [2.]]);
        let b = tensor!([2.]);
        assert_eq!(a.matvec(b), tensor!([2., 4.]));

        let a = tensor!([[1., 2.], [2., 3.]]);
        let b = tensor!([2., 3.]);
        assert_eq!(a.matvec(b), tensor!([8., 13.]));
    }

    #[test]
    fn test_matvec_3_n() {
        let a = tensor!([[1.], [2.], [3.]]);
        let b = tensor!([2.]);
        assert_eq!(a.matvec(b), tensor!([2., 4., 6.]));
    }
}
