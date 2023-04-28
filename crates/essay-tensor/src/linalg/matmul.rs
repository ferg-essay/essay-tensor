use crate::tensor::{Tensor, Op, BoxOp, TensorUninit};

enum Transpose {
    None,
    TransposeA,
    TransposeB,
}

#[derive(Debug, Clone)]
struct Matmul;

impl Op for Matmul {
    fn box_clone(&self) -> BoxOp {
        Box::new(Matmul)
    }
}

impl<const N:usize> Tensor<N> {
    pub fn matmul<const M:usize>(&self, b: Tensor<M>) -> Tensor<M> {
        assert_eq!(M, N, "matrix multiplication requires matching dim >= 2");
        assert!(N > 1, "matrix multiplication requires dim >= 2");
        assert_eq!(&self.shape()[2..], &b.shape()[2..], "matmul batch shape must match");
        assert_eq!(self.shape()[0], b.shape()[1], "matmul a.shape[0] must equal b.shape[1]");

        let n : usize = self.shape()[2..].iter().product();
        let a_size = self.shape()[0] * self.shape()[1];
        let b_size = b.shape()[0] * b.shape()[1];
        let o_size = self.shape()[1] * b.shape()[0];

        unsafe {
            let mut out = TensorUninit::new(o_size * n);

            let mut a_start = 0;
            let mut b_start = 0;
            let mut o_start = 0;

            let cols = b.shape()[0];
            let rows = self.shape()[1];
    
            for _ in 0..n {
                naive_matmul(
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
            o_shape[0] = b.shape()[0];
            o_shape[1] = self.shape()[1];
            // Tensor::new(Rc::new(out), o_shape)
            self.next_binop(&b, out.init(), o_shape, Matmul.box_clone())
        }
    }
}

unsafe fn naive_matmul<const M:usize, const N:usize>(
    out: &mut TensorUninit, 
    out_start: usize,
    a: &Tensor<N>, 
    a_start: usize,
    b: &Tensor<M>,
    b_start: usize,
    cols: usize,
    rows: usize,
) {
    let out_stride = cols;
    let a_stride = a.shape()[0];
    let b_stride = b.shape()[0];

    let a_ptr = a.buffer().as_ptr();
    let b_ptr = b.buffer().as_ptr();
    let out_ptr = out.as_ptr();

    let mut out_row = out_start;
    let mut a_row = a_start;

    for _ in 0..rows {
        for col in 0..cols {
            let mut b_off = b_start + col;
            let mut v: f32 = 0.;
            for k in 0..a_stride {
                v += a_ptr.add(a_row + k).read() * b_ptr.add(b_off).read();

                b_off += b_stride;
            }

            out_ptr.add(out_row + col).write(v);
        }

        out_row += out_stride;
        a_row += a_stride;
    }

}

#[cfg(test)]
mod test {
    use crate::{tensor, Tensor};

    //use super::matmul;

    #[test]
    fn test_matmul_1() {
        let a = tensor!([[2.]]);
        let b = tensor!([[3.]]);

        assert_eq!(a.matmul(b), tensor!([[6.]]));
    }

    #[test]
    fn test_matmul_vectors() {
        let a = tensor!([[1., 2.]]);
        let b = tensor!([[1.], [3.]]);
        assert_eq!(a.matmul(b), tensor!([[7.]]));

        let a = tensor!([[1., 2.]]);
        let b = tensor!([[1.], [3.]]);
        assert_eq!(b.matmul(a), tensor!([[1., 2.], [3., 6.]]));
    }

    #[test]
    fn test_matmul_square() {

        let id = tensor!([[1., 0.], [0., 1.]]);
        assert_eq!(&id.clone().matmul(id.clone()), &id);

        let a = tensor!([[1., 0.], [0., 2.]]);
        assert_eq!(&id.clone().matmul(a.clone()), &a);
        assert_eq!(&a.clone().matmul(id.clone()), &a);
        assert_eq!(&a.clone().matmul(a.clone()),
            &tensor!([[1., 0.], [0., 4.]]));

        let top = tensor!([[1., 1.], [0., 1.]]);
        assert_eq!(top.clone().matmul(top.clone()), tensor!([[1., 2.], [0., 1.]]));

        let bot = tensor!([[1., 0.], [1., 1.]]);
        assert_eq!(bot.clone().matmul(bot.clone()), tensor!([[1., 0.], [2., 1.]]));
    }

    #[test]
    fn test_matmul_2x3() {
        let a = tensor!([[1., 0., 2.], [0., 1., 10.]]);
        let b = tensor!([[1., 0.], [0., 1.], [3., 4.]]);
        assert_eq!(a.clone().matmul(b.clone()), tensor!([[7., 8.], [30., 41.]]));
        assert_eq!(b.clone().matmul(a.clone()), tensor!([
            [1., 0., 2.],
            [0., 1., 10.],
            [3., 4., 46.]]));
    }
}
