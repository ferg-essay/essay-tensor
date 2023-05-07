use crate::{tensor::{Tensor, TensorUninit}, model::{EvalOp}};

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
        a_cols: usize, 
        a_rows: usize, 
        b_cols: usize, 
        b_rows: usize
    ) -> usize;

    fn out_rows(&self, a_cols: usize, a_rows: usize) -> usize;
    fn out_cols(&self, b_cols: usize, b_rows: usize) -> usize;

    fn a_inner_stride(&self, a_cols: usize, a_rows: usize) -> usize;
    fn a_outer_stride(&self, a_cols: usize, a_rows: usize) -> usize;

    fn b_inner_stride(&self, b_cols: usize, b_rows: usize) -> usize;
    fn b_outer_stride(&self, b_cols: usize, b_rows: usize) -> usize;
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

    let (a_cols, a_rows) = (a.shape()[0], a.shape()[1]);
    let (b_cols, b_rows) = (b.shape()[0], b.shape()[1]);

    let o_cols = transpose.out_cols(b_cols, b_rows);
    let o_rows = transpose.out_rows(a_cols, a_rows);

    let inner_len = transpose.inner_len(a_cols, a_rows, b_cols, b_rows);

    let a_inc = transpose.a_inner_stride(a_cols, a_rows);
    let a_stride = transpose.a_outer_stride(a_cols, a_rows);

    let b_inc = transpose.b_inner_stride(b_cols, b_rows);
    let b_stride = transpose.b_outer_stride(b_cols, b_rows);

    let batch_len : usize = a.shape()[2..].iter().product();
    let a_size = a_rows * a_cols;
    let b_size = b_rows * b_cols;
    let o_size = o_rows * o_cols;

    unsafe {
        let out = TensorUninit::<f32>::new(o_size * batch_len);

        for batch in 0..batch_len {
            let a_ptr = a.data().as_ptr().add(a_size * batch);
            let b_ptr = b.data().as_ptr().add(b_size * batch);
            let out_ptr = out.as_ptr().add(o_size * batch);
        
            naive_matmul(
                out_ptr,
                o_cols,
                o_rows,
                inner_len,
                a_ptr,
                a_inc,
                a_stride,
                b_ptr,
                b_inc,
                b_stride,
            );
        }

        let mut o_shape = b.shape().clone();
        o_shape[0] = o_cols;
        o_shape[1] = o_rows;

        a.next_binop(&b, out.init(), o_shape, Matmul)
    }
}

unsafe fn naive_matmul(
    out_ptr: *mut f32,
    o_cols: usize,
    o_rows: usize,
    inner_len: usize,
    a_ptr: *const f32,
    a_inc: usize,
    a_stride: usize,
    b_ptr: *const f32,
    b_inc: usize,
    b_stride: usize,
) {
    for row in 0..o_rows {
        let a_row = row * a_stride;

        for col in 0..o_cols {
            let b_col = col * b_stride;

            let mut v: f32 = 0.;
            for k in 0..inner_len {
                v += a_ptr.add(a_row + k * a_inc).read() 
                    * b_ptr.add(b_col + k * b_inc).read();
            }

            out_ptr.add(row * o_cols + col).write(v);
        }
    }
}

impl TransposeMatmul for Transpose {
    #[inline]
    fn inner_len(
        &self, 
        a_cols: usize, 
        a_rows: usize,
        b_cols: usize,
        b_rows: usize,
    ) -> usize {
        match self {
            Transpose::None => {
                assert_eq!(a_cols, b_rows, "left columns must match right rows");
                a_cols
            },
            Transpose::TransposeA => {
                assert_eq!(a_rows, b_rows, "left rows must match right rows {:?}", &self);
                a_rows
            },
            Transpose::TransposeB => {
                assert_eq!(a_cols, b_cols, "left columns must match right columns {:?}", &self);
                a_cols
            },
            Transpose::TransposeAB => {
                assert_eq!(a_rows, b_cols, "left rows must match right columns {:?}", &self);
                a_rows
            },
        }
    }

    #[inline]
    fn out_cols(&self, b_cols: usize, b_rows: usize) -> usize {
        match self {
            Transpose::None => b_cols,
            Transpose::TransposeA => b_cols,
            Transpose::TransposeB => b_rows,
            Transpose::TransposeAB => b_rows,
        }
    }

    #[inline]
    fn out_rows(&self, a_cols: usize, a_rows: usize) -> usize {
        match self {
            Transpose::None => a_rows,
            Transpose::TransposeA => a_cols,
            Transpose::TransposeB => a_rows,
            Transpose::TransposeAB => a_rows,
        }
    }

    fn a_inner_stride(&self, a_cols: usize, _a_rows: usize) -> usize
    {
        match self {
            Transpose::None => 1,
            Transpose::TransposeA => a_cols,
            Transpose::TransposeB => 1,
            Transpose::TransposeAB => 1,
        }
    }

    fn a_outer_stride(&self, a_cols: usize, _a_rows: usize) -> usize
    {
        match self {
            Transpose::None => a_cols,
            Transpose::TransposeA => 1,
            Transpose::TransposeB => a_cols,
            Transpose::TransposeAB => a_cols,
        }
    }

    fn b_inner_stride(&self, b_cols: usize, _b_rows: usize) -> usize
    {
        match self {
            Transpose::None => b_cols,
            Transpose::TransposeA => b_cols,
            Transpose::TransposeB => 1,
            Transpose::TransposeAB => 1,
        }
    }

    fn b_outer_stride(&self, b_cols: usize, _b_rows: usize) -> usize
    {
        match self {
            Transpose::None => 1,
            Transpose::TransposeA => 1,
            Transpose::TransposeB => b_cols,
            Transpose::TransposeAB => b_cols,
        }
    }
}

impl EvalOp for Matmul {
    fn eval(
        &self,
        _tensors: &crate::model::TensorCache,
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
