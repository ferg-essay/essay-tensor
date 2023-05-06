use std::{ops, sync::Arc};

use crate::{tensor::{Tensor, TensorUninit}, model::{BackOp, BoxForwardOp}};

use super::matmul::Transpose;

#[derive(Debug, Clone)]
struct Matvec;

pub trait TransposeMatvec {
    fn inner_len(
        &self, 
        b_cols: usize, 
        b_rows: usize
    ) -> usize;

    fn a_inc(&self, a_cols: usize, a_rows: usize) -> usize;
    fn a_stride(&self, a_cols: usize, a_rows: usize) -> usize;
    fn a_size(&self, a_cols: usize, a_rows: usize) -> usize;

    fn b_inc(&self, b_cols: usize, b_rows: usize) -> usize;

    fn out_size(&self, a_cols: usize, a_rows: usize) -> usize;

}

impl TransposeMatvec for Transpose {
    fn inner_len(
        &self, 
        b_cols: usize, 
        b_rows: usize
    ) -> usize {
        match self {
            Transpose::None => b_cols,
            Transpose::TransposeA => b_cols,
            Transpose::TransposeB => todo!(),
        }
    }

    fn a_inc(&self, a_cols: usize, _a_rows: usize) -> usize {
        match self {
            Transpose::None => 1,
            Transpose::TransposeA => a_cols,
            Transpose::TransposeB => todo!(),
        }
    }

    fn a_stride(&self, a_cols: usize, _a_rows: usize) -> usize {
        match self {
            Transpose::None => a_cols,
            Transpose::TransposeA => 1,
            Transpose::TransposeB => todo!(),
        }
    }

    fn a_size(&self, a_cols: usize, a_rows: usize) -> usize {
        match self {
            Transpose::None => a_rows,
            Transpose::TransposeA => a_cols,
            Transpose::TransposeB => todo!(),
        }
    }

    fn b_inc(&self, b_cols: usize, b_rows: usize) -> usize {
        match self {
            Transpose::None => b_cols,
            Transpose::TransposeA => b_cols,
            Transpose::TransposeB => todo!(),
        }
    }

    fn out_size(&self, a_cols: usize, a_rows: usize) -> usize {
        match self {
            Transpose::None => a_rows,
            Transpose::TransposeA => a_cols,
            Transpose::TransposeB => todo!(),
        }
    }
}

impl Tensor<f32> {
    pub fn matvec(&self, b: &Tensor<f32>) -> Tensor {
        matvec_t(&self, b, Transpose::None)
    }

    pub fn matvec_t(&self, b: &Tensor<f32>, transpose: impl TransposeMatvec) -> Tensor {
        matvec_t(&self, b, transpose)
    }
}

pub fn matvec(a: &Tensor<f32>, b: &Tensor<f32>) -> Tensor {
    matvec_t(a, b, Transpose::None)
}

pub fn matvec_t(
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    transpose: impl TransposeMatvec,
) -> Tensor<f32> {
    assert!(a.rank() >= 2, "matrix[{}]-vector multiplication requires dim >= 2", a.rank());
    
    assert_eq!(a.shape()[2..], b.shape()[1..], "matvec batch shape must match");

    let n : usize = a.shape()[2..].iter().product();

    let (a_cols, a_rows) = (a.shape()[0], a.shape()[1]);
    let b_cols = b.shape()[0];

    let inner_len = b_cols;

    let a_inc = transpose.a_inc(a_cols, a_rows);
    let a_stride = transpose.a_stride(a_cols, a_rows);
    let a_size = transpose.a_size(a_cols, a_rows);

    let o_size = transpose.out_size(a_cols, a_rows);

    unsafe {
        let mut out = TensorUninit::<f32>::new(o_size * n);

        for block in 0..n {
            naive_matvec_f32(
                &mut out, 
                block * o_size,
                &a,
                block * a_size,
                a_inc,
                a_stride,
                a_size,
                &b,
                block * b_cols,
                b_cols,
                o_size,
            );
        }

        let mut o_shape = b.shape().clone();
        o_shape[0] = o_size;
        // Tensor::new(Rc::new(out), o_shape)
        // a.next_binop(&b, out.init(), o_shape, Matvec.box_clone())
        todo!()
    }
}

unsafe fn naive_matvec_f32(
    out: &mut TensorUninit<f32>, 
    out_start: usize,
    a: &Tensor<f32>, 
    a_start: usize,
    a_inc: usize,
    a_stride: usize,
    a_size: usize,
    b: &Tensor<f32>,
    b_start: usize,
    b_size: usize,
    o_cols: usize,
) {
    let a_stride = a.shape()[0];

    let a_data = a.buffer();
    let b_data = b.buffer();

    let mut a_row = a_start;

    for col in 0..o_cols {
        let mut len = a_stride;

        let mut a_off = a_start + col * a_stride;
        let mut b_off = b_start;

        let mut v = 0.0;

        // unroll for simd
        while len > 4 {
            let a_chunk = a_data.read_4(a_off, a_inc);
            let b_chunk = b_data.read_4(b_off, 1);

            v += &a_chunk.muladd(&b_chunk);

            a_off += 4 * a_inc;
            b_off += 4;
            len -= 4;
        }

        for i in 0..len {
            v += a_data.get_unchecked(a_off + i * a_inc)
                * b_data.get_unchecked(b_off + i * a_inc);
        }

        out.set_unchecked(out_start + col, v);
        a_row += a_stride;
    }
}

impl BackOp for Matvec {
    fn gradient(
            &self, 
            i: usize, 
            args: &[&Tensor],
            prev: &Option<Tensor>, 
        ) -> Tensor {
            println!("args[0] {:?}", args[0]);
            println!("args[1] {:?}", args[1]);
            println!("ones[0] {:?}", Tensor::ones(args[0].shape()));
            println!("prev {:?}", prev);
        match prev {
            None => {
                if i == 0 {
                    Tensor::ones(args[0].shape()) * args[1]
                } else {
                    matvec_0_gradient(args[0])
                }
            },
            Some(prev) => {
                if i == 0 {
                    Tensor::ones(args[0].shape())
                } else {
                    println!("Prev {:?}", prev);
                    println!("Args[0] {:?}", args[0]);
                    args[0].matvec(prev)
                }
            },
        }
    }

    /*
    fn box_clone(&self) -> BoxForwardOp {
        //Box::new(Matvec)
        todo!()
    }
    */
}

fn matvec_0_gradient(a: &Tensor) -> Tensor {
    let o_cols = a.shape()[0];

    let a_rows = a.shape()[1];

    unsafe {
        let mut out = TensorUninit::<f32>::new(o_cols);
        let a_data = a.buffer();

        for i in 0..o_cols {
            let mut v = 0.;
            for j in 0..a_rows {
                v += a_data.get_unchecked(j * o_cols + i);
            }
            out.set_unchecked(i, v);
        }

        let mut shape: Vec::<usize> =  a.shape().iter().skip(1).map(|s| *s).collect();
        shape[0] = o_cols;

        Tensor::new(Arc::new(out.init()), &shape)
    }
}

#[cfg(test)]
mod test {
    use crate::{tensor, Tensor, model::{Var, Tape}};

    #[test]
    fn test_matvec_1_1() {
        let a = tensor!([[2.]]);
        let b = tensor!([3.]);

        assert_eq!(a.matvec(&b), tensor!([6.]));
    }

    #[test]
    fn test_matvec_1_2() {
        let a = tensor!([[1., 2.]]);
        let b = tensor!([3., 4.]);

        assert_eq!(a.matvec(&b), tensor!([11.]));
    }

    #[test]
    fn test_matvec_2_n() {
        let a = tensor!([[1.], [2.]]);
        let b = tensor!([2.]);
        assert_eq!(a.matvec(&b), tensor!([2., 4.]));

        let a = tensor!([[1., 2.], [2., 3.]]);
        let b = tensor!([2., 3.]);
        assert_eq!(a.matvec(&b), tensor!([8., 13.]));
    }

    #[test]
    fn test_matvec_3_n() {
        let a = tensor!([[1.], [2.], [3.]]);
        let b = tensor!([2.]);
        assert_eq!(a.matvec(&b), tensor!([2., 4., 6.]));
    }

    #[test]
    fn test_mul_2_1() {
        let a = tensor!([[1.], [2.], [3.]]);
        let b = tensor!([2.]);
        assert_eq!(a * b, tensor!([2., 4., 6.]));
    }
    
    #[test]
    fn backprop_1_1() {
        let a = Var::new("a", tensor!([[1.]]));
        let x = Var::new("x", tensor!([1.]));
    
        let mut tape = Tape::with(|| {
            let loss: Tensor = a.matvec(&x);
    
            Ok(loss)
        }).unwrap();
    
        let dz = tape.gradient(&a);
        assert_eq!(dz, tensor!([[1.0]]));

        let a = Var::new("a", tensor!([[1., 2., 3.], [4., 5., 6.]]));
        let x = Var::new("x", tensor!([10., 20., 30.]));
    
        let mut tape = Tape::with(|| {
            let loss: Tensor = a.matvec(&x);
    
            Ok(loss)
        }).unwrap();
    
        let dz = tape.gradient(&a);
        assert_eq!(dz, tensor!([[10., 20., 30.], [10., 20., 30.]]));

        // gradient

        let a = Var::new("a", tensor!([[1.]]));
        let x = Var::new("x", tensor!([1.]));
    
        let mut tape = Tape::with(|| {
            let loss: Tensor = a.matvec(&x);
    
            Ok(loss)
        }).unwrap();
    
        let dz = tape.gradient(&x);
        assert_eq!(dz, tensor!([1.0]));

        let a = Var::new("a", tensor!([[1., 2., 3.], [4., 5., 6.]]));
        let x = Var::new("x", tensor!([10., 20., 30.]));
    
        let mut tape = Tape::with(|| {
            let out: Tensor = a.matvec(&x);
            assert_eq!(out, tensor!([140., 320.]));
    
            Ok(out)
        }).unwrap();
    
        let dz = tape.gradient(&x);
        assert_eq!(dz, tensor!([5., 7., 9.]));
    }
    
    #[test]
    fn backprop_1_1_prev() {
        let a = Var::new("a", tensor!([[1.]]));
        let x = Var::new("x", tensor!([1.]));
    
        let mut tape = Tape::with(|| {
            let out: Tensor = a.matvec(&x);

            let loss = out.l2_loss();

            println!("Loss: {:?}", loss);
    
            Ok(loss)
        }).unwrap();
    
        let dz = tape.gradient(&x);
        assert_eq!(dz, tensor!([1.0]));

        let a = Var::new("a", tensor!([[1., 2., 3.], [4., 5., 6.]]));
        let x = Var::new("x", tensor!([10., 20., 30.]));
    
        let mut tape = Tape::with(|| {
            let out: Tensor = a.matvec(&x);

            Ok(out.l2_loss())
        }).unwrap();

        //let dz = tape.gradient(&a);
        //assert_eq!(dz, tensor!([10., 20., 30.]));

        let dz = tape.gradient(&x);
        assert_eq!(dz, tensor!([1420., 1880., 2340.]));

        // gradient

        let a = Var::new("a", tensor!([[1.]]));
        let x = Var::new("x", tensor!([1.]));
    
        let mut tape = Tape::with(|| {
            let out: Tensor = a.matvec(&x);
            let loss = out.l2_loss();

            Ok(loss)
        }).unwrap();
    
        let dz = tape.gradient(&x);
        assert_eq!(dz, tensor!([1.0]));

        let a = Var::new("a", tensor!([[1., 2., 3.], [4., 5., 6.]]));
        let x = Var::new("x", tensor!([10., 20., 30.]));
    
        let mut tape = Tape::with(|| {
            let out: Tensor = a.matvec(&x);
            assert_eq!(out, tensor!([140., 320.]));
    
            Ok(out.l2_loss())
        }).unwrap();
    
        let dz = tape.gradient(&a);
        assert_eq!(dz, tensor!([[1400., 2800., 4200.], [3200., 6400., 9600.]]));
    
        let dz = tape.gradient(&x);
        assert_eq!(dz, tensor!([1420., 1880., 2340.]));
    }
}
