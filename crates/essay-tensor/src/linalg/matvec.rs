use std::{any::type_name, cmp};

use crate::{
    tensor::{Tensor, TensorId, TensorUninit}, 
    model::{Operation, Expr, expr::GradientOp, Tape, NodeOp, IntoForward}, linalg::blas::sgemm
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
    let node = NodeOp::new(&[a, b], Matvec.to_op());

    let value = matvec_t_op(a, b, Transpose::None, node);

    Tape::set_tensor(value)
}

pub fn matvec_t(
    a: &Tensor<f32>,
    x: &Tensor<f32>,
    _transpose: impl TransposeMatvec,
) -> Tensor<f32> {
    let node = NodeOp::new(&[a, x], Matvec.to_op());

    let value = matvec_t_op(a, x, _transpose, node);

    Tape::set_tensor(value)
}

fn matvec_t_op(
    a: &Tensor<f32>,
    x: &Tensor<f32>,
    transpose: impl TransposeMatvec,
    id: TensorId,
) -> Tensor<f32> {
    assert!(a.rank() >= 2, "matrix[{}]-vector multiplication requires dim >= 2", a.rank());
    
    let batch : usize = a.broadcast_min(2, &x, cmp::min(2, x.rank()));

    let a_size = a.cols() * a.rows();

    let x_rows = cmp::max(1, x.rows());
    let x_size = x.cols() * x_rows;

    let (o_cols, o_rows) = transpose.o_size(a, x);
    let o_size = o_cols * o_rows;

    unsafe {
        let mut out = TensorUninit::<f32>::new(o_size * batch);

        for n in 0..batch {
            let a_ptr = a.as_wrap_ptr(n * a_size);
            let x_ptr = x.as_wrap_ptr(n * x_size);
            let o_ptr = out.as_mut_ptr().add(n * o_size);

            transpose.sgemm(a, x, o_cols, o_rows, a_ptr, x_ptr, o_ptr);
        }

        let mut o_shape = Vec::from(x.shape().as_slice());
        let len = o_shape.len();
        o_shape[len - 1] = o_cols;
        let tensor = Tensor::from_uninit_with_id(out, o_shape, id);

        // Tape::set_tensor(tensor)
        tensor
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
                    &a.shape().as_slice(), &x.shape().as_slice());

                (a.rows(), cmp::max(1, x.rows()))
            },
            Transpose::TransposeA => {
                assert_eq!(a.rows(), x.cols(),
                    "matvec shapes do not match. A={:?} x={:?} for {:?}",
                    &a.shape().as_slice(), &x.shape().as_slice(), self);
                    
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

impl Operation for Matvec {
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

    fn df(
        &self,
        _forward: &Expr,
        graph: &mut Expr,
        i: usize,
        args: &[TensorId],
        prev: TensorId,
    ) -> TensorId {
        match i {
            0 => {
                graph.add_grad_op(MatvecBackLeft, &[args[1]], prev)
            },
            1 => {
                // let left = graph.constant_id(args[0]);
                graph.add_grad_op(MatvecBackRight, &[args[0]], prev)
            }
            _ => panic!("invalid argument")
        }
    }
}

impl GradientOp for MatvecBackLeft {
    fn name(&self) -> &str {
        type_name::<Self>()
    }
    
    fn df(
        &self,
        args: &[&Tensor],
        prev: &Tensor,
    ) -> Tensor {
        let mut x = args[0].clone();
        let mut prev = prev.clone();

        if x.rank() <= 1 {
            x = x.reshape([1, x.cols()]);
        }
        if prev.rank() <= 1 {
            prev = prev.reshape([1, prev.cols()]);
        }
        //args[0].outer_product(prev)
        // TODO: matmul_t_op
        x.matmul_t(&prev, Transpose::TransposeA)
    }
}

impl GradientOp for MatvecBackRight {
    fn name(&self) -> &str {
        type_name::<Self>()
    }
    
    fn df(
        &self,
        args: &[&Tensor],
        prev: &Tensor,
    ) -> Tensor {
        args[0].matvec_t(prev, Transpose::TransposeA)
    }
}

#[cfg(test)]
mod test {
    use crate::{tensor, Tensor, model::{Var, Trainer}, linalg::matmul::Transpose, tf32};

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
    fn test_matvec_t() {
        let a = tensor!([[1., 4.], [2., 5.], [3., 6.]]);
        let b = tensor!([10., 20.]);
        assert_eq!(a.matvec(&b), tensor!([90., 120., 150.]));

        let a = tensor!([[1., 2., 3.], [4., 5., 6.]]);
        let b = tensor!([10., 20.]);
        assert_eq!(a.matvec_t(&b, Transpose::TransposeA), tensor!([90., 120., 150.]));
    }

    #[test]
    fn matvec_1x1_by_1() {
        // assert_eq!(a.matvec(&b), tensor!([2., 20.]));
        
        let a = tf32!([[10.]]);
        let x = tf32!([2.]);

        assert_eq!(a.matvec(&x), tf32!([20.]));

        let a = Var::new("a", tf32!([[10.]]));
        let x = Var::new("x", tf32!([2.]));

        let module = Trainer::compile((), |()| {
            a.matvec(&x)
        }); // .training(&[&a, &x]);
        let train = module.train(());

        assert_eq!(train.value(), tensor!([20.]));
        assert_eq!(train.gradient(&a), tensor!([[2.0]]));
        assert_eq!(train.gradient(&x), tensor!([10.0]));
    }

    #[test]
    fn matvec_1x1_by_1x1() {
        // assert_eq!(a.matvec(&b), tensor!([2., 20.]));
        
        let a = tf32!([[10.]]);
        let x = tf32!([[2.]]);

        assert_eq!(a.matvec(&x), tf32!([[20.]]));

        let a = Var::new("a", tf32!([[10.]]));
        let x = Var::new("x", tf32!([[2.]]));

        let module = Trainer::compile((), |()| {
            a.matvec(&x)
        }); // .training(&[&a, &x]);
        let train = module.train(());

        assert_eq!(train.value(), tensor!([[20.]]));
        assert_eq!(train.gradient(&a), tensor!([[2.0]]));
        assert_eq!(train.gradient(&x), tensor!([[10.0]]));
    }

    #[test]
    fn matvec_1x1_by_1x2() {
        let a = tf32!([[10.]]);
        let x = tf32!([[1.], [2.]]);

        assert_eq!(a.matvec(&x), tf32!([[10.], [20.]]));
        assert_eq!(
            a.matvec_t(&tf32!([[1.], [1.]]), Transpose::TransposeA),
            tensor!([[10.], [10.]])
        );

        let a = Var::new("a", tf32!([[10.]]));
        let x = Var::new("x", tf32!([[1.], [2.]]));

        let module = Trainer::compile((), |()| {
            a.matvec(&x)
        }); // .training(&[&a, &x]);
        let train = module.train(());

        assert_eq!(train.value(), tensor!([[10.], [20.]]));
        assert_eq!(train.gradient(&a), tensor!([[3.0]]));
        assert_eq!(train.gradient(&x), tensor!([[10.0], [10.0]]));
    }

    #[test]
    fn matvec_1x1_by_2x1x1() {
        let a = tf32!([[10.]]);
        let x = tf32!([[[2.]], [[3.]]]);

        assert_eq!(a.matvec(&x), tf32!([[[20.]], [[30.]]]));

        let a = Var::new("a", tf32!([[10.]]));
        let x = Var::new("x", tf32!([[[2.]], [[3.]]]));

        let module = Trainer::compile((), |()| {
            a.matvec(&x)
        });
        let train = module.train(());

        assert_eq!(train.value(), tensor!([[[20.]], [[30.]]]));
        assert_eq!(train.gradient(&a), tensor!([[5.0]]));
        assert_eq!(train.gradient(&x), tensor!([[[10.0]], [[10.0]]]));
    }

    #[test]
    fn matvec_2x1_by_1() {
        // assert_eq!(a.matvec(&b), tensor!([2., 20.]));
        
        let a = tf32!([[1.], [10.]]);
        let x = tf32!([2.]);

        assert_eq!(a.matvec(&x), tensor!([2., 20.]));
        assert_eq!(
            a.matvec_t(&tf32!([1., 1.]), Transpose::TransposeA),
            tensor!([11.])
        );

        let a = Var::new("a", tf32!([[1.], [10.]]));
        let x = Var::new("x", tf32!([2.]));

        let module = Trainer::compile((), |()| {
            a.matvec(&x)
        }); // .training(&[&a, &x]);
        let train = module.train(());

        assert_eq!(train.value(), tensor!([2., 20.]));
        assert_eq!(train.gradient(&x), tensor!([11.0]));
        assert_eq!(train.gradient(&a), tensor!([[2.0], [2.]]));
    }

    #[test]
    fn matvec_1x2_by_2() {
        // assert_eq!(a.matvec(&b), tensor!([2., 20.]));
        
        let a = tf32!([[10., 20.]]);
        let x = tf32!([1., 3.]);

        assert_eq!(a.matvec(&x), tf32!([70.]));

        let a = Var::new("a", tf32!([[10., 20.]]));
        let x = Var::new("x", tf32!([1., 3.]));

        let module = Trainer::compile((), |()| {
            a.matvec(&x)
        });
        let train = module.train(());

        assert_eq!(train.value(), tensor!([70.]));
        assert_eq!(train.gradient(&x), tensor!([10.0, 20.0]));
        assert_eq!(train.gradient(&a), tensor!([[1.0, 3.0]]));
    }

    #[test]
    #[should_panic]
    fn matvec_2x1_by_2() {
        // assert_eq!(a.matvec(&b), tensor!([2., 20.]));
        
        let a = tf32!([[10.], [20.]]);
        let x = tf32!([1., 3.]);

        assert_eq!(a.matvec(&x), tf32!([[10.], [20.]]));
    }
    
    #[test]
    fn backprop_1_1() {
        let a = Var::new("a", tensor!([[1.]]));
        let x = Var::new("x", tensor!([1.]));
    
        let module = Trainer::compile((), |()| {
            a.matvec(&x)
        }); // .training(&[&a, &x]);
        let train = module.train(());

        assert_eq!(train.value(), tensor!([1.]));
        assert_eq!(train.gradient(&a), tensor!([[1.0]]));
        assert_eq!(train.gradient(&x), tensor!([1.0]));

        let a = Var::new("a", tensor!([[1., 2., 3.], [4., 5., 6.]]));
        let x = Var::new("x", tensor!([10., 20., 30.]));
    
        let module = Trainer::compile((), |()| {
            a.matvec(&x)
        }); // .training(&[&a, &x]);
        let train = module.train(());

        assert_eq!(train.value(), tensor!([140., 320.]));
        assert_eq!(train.gradient(&a), tensor!([[10., 20., 30.], [10., 20., 30.]]));
        assert_eq!(train.gradient(&x), tensor!([5., 7., 9.]));
    }
    
    #[test]
    fn backprop_1_1_prev() {
        let a = Var::new("a", tensor!([[1.]]));
        let x = Var::new("x", tensor!([1.]));
    
        let module = Trainer::compile((), |()| {
            let out: Tensor = a.matvec(&x);

            out.l2_loss()
        }); // .training(&[&a, &x]);
        let train = module.train(());

        assert_eq!(train.value(), tensor!(0.5));
        assert_eq!(train.gradient(&a), tensor!([[1.0]]));
        assert_eq!(train.gradient(&x), tensor!([1.0]));

        let a = Var::new("a", tensor!([[1., 2., 3.], [4., 5., 6.]]));
        let x = Var::new("x", tensor!([10., 20., 30.]));
    
        let trainer = Trainer::compile((), |()| {
            let out = a.matvec(&x);
            assert_eq!(out, tensor!([140., 320.]));

            out.l2_loss()
        }); // .training(&[&a, &x]);
        let train = trainer.train(());
    
        let da = train.gradient(&a);
        assert_eq!(da, tensor!([[1400., 2800., 4200.], [3200., 6400., 9600.]]));
    
        let dx = train.gradient(&x);
        assert_eq!(dx, tensor!([1420., 1880., 2340.]));
    }
}
