use std::{any::type_name};

use crate::{
    tensor::{Tensor, TensorId, TensorUninit, NodeId}, 
    function::{Operation, Graph, graph::BackOp}, linalg::blas::sgemm
};

use super::matmul::Transpose;

#[derive(Debug, Clone)]
struct Matvec;

#[derive(Debug, Clone)]
struct MatvecOuter;

#[derive(Debug, Clone)]
struct MatvecBackRight;

#[derive(Debug, Clone)]
struct MatvecBackRightT;

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
    x: &Tensor<f32>,
    _transpose: impl TransposeMatvec,
) -> Tensor<f32> {
    assert!(a.rank() >= 2, "matrix[{}]-vector multiplication requires dim >= 2", a.rank());
    
    //assert_eq!(a.shape().as_subslice(2..), x.shape().as_subslice(1..), "matvec batch shape must match");
    assert_eq!(x.cols(), a.cols());

    let n : usize = a.broadcast_min(2, &x, 2);

    let a_dim = [a.cols(), a.rows()];
    let x_dim = [x.cols(), x.rows()];

    let o_size = a_dim[1] * x_dim[1];

    let a_size = a_dim[0] * a_dim[1];
    let x_size = x_dim[0] * x_dim[1];

    unsafe {
        let mut out = TensorUninit::<f32>::new(o_size * n);

        for block in 0..n {
            sgemm(
                x.rows(), x_dim[0], o_size,
                1.,
                x.as_wrap_ptr(block * x_size),
                x_dim[0], 1,
                a.as_wrap_ptr(block * a_size),
                1, a_dim[0],
                0.,
                out.as_mut_ptr().add(block * o_size),
                o_size, 1,
            )
        }

        let mut o_shape = Vec::from(x.shape().as_slice());
        let len = o_shape.len();
        o_shape[len - 1] = o_size;
        Tensor::from_uninit(out, o_shape)
        //a.next_binop(&b, out.init(), o_shape, Matvec)
        //todo!()
    }
}

impl TransposeMatvec for Transpose {
    fn inner_len(
        &self, 
        b_cols: usize, 
        _b_rows: usize
    ) -> usize {
        match self {
            Transpose::None => b_cols,
            Transpose::TransposeA => b_cols,
            Transpose::TransposeB => todo!(),
            Transpose::TransposeAB => todo!(),
        }
    }

    fn a_inc(&self, a_cols: usize, _a_rows: usize) -> usize {
        match self {
            Transpose::None => 1,
            Transpose::TransposeA => a_cols,
            Transpose::TransposeB => todo!(),
            Transpose::TransposeAB => todo!(),
        }
    }

    fn a_stride(&self, a_cols: usize, _a_rows: usize) -> usize {
        match self {
            Transpose::None => a_cols,
            Transpose::TransposeA => 1,
            Transpose::TransposeB => todo!(),
            Transpose::TransposeAB => todo!(),
        }
    }

    fn a_size(&self, a_cols: usize, a_rows: usize) -> usize {
        match self {
            Transpose::None => a_rows,
            Transpose::TransposeA => a_cols,
            Transpose::TransposeB => todo!(),
            Transpose::TransposeAB => todo!(),
        }
    }

    fn b_inc(&self, b_cols: usize, _b_rows: usize) -> usize {
        match self {
            Transpose::None => b_cols,
            Transpose::TransposeA => b_cols,
            Transpose::TransposeB => todo!(),
            Transpose::TransposeAB => todo!(),
        }
    }

    fn out_size(&self, a_cols: usize, a_rows: usize) -> usize {
        match self {
            Transpose::None => a_rows,
            Transpose::TransposeA => a_cols,
            Transpose::TransposeB => todo!(),
            Transpose::TransposeAB => todo!(),
        }
    }
}

impl Operation for Matvec {
    fn name(&self) -> &str {
        type_name::<Self>()
    }
    
    fn forward(
        &self,
        args: &[&Tensor],
        _node: NodeId,
    ) -> Tensor {
        matvec(args[0], args[1])
    }

    fn back(
        &self,
        _forward: &Graph,
        graph: &mut Graph,
        i: usize,
        args: &[TensorId],
        prev: TensorId,
    ) -> TensorId {
        match i {
            0 => {
                graph.add_back_op(MatvecOuter, &[args[1]], prev)
            },
            1 => {
                // let left = graph.constant_id(args[0]);
                graph.add_back_op(MatvecBackRightT, &[args[0]], prev)
            }
            _ => panic!("invalid argument")
        }
    }
}

impl BackOp for MatvecOuter {
    fn name(&self) -> &str {
        type_name::<Self>()
    }
    
    fn df(
        &self,
        args: &[&Tensor],
        prev: &Tensor,
    ) -> Tensor {
        args[0].outer_product(prev)
    }
}

impl BackOp for MatvecBackRightT {
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
    use crate::{tensor, Tensor, function::{Var, Trainer}, linalg::matmul::Transpose};

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
