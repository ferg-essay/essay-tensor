use std::{any::type_name, cmp};

use crate::{model::{Expr, Operation, IntoForward, NodeOp, Tape, expr::GradientOp}, Tensor, 
    tensor::{Dtype, TensorId, TensorUninit}, prelude::Shape
};

use super::UnaryKernel;

pub trait BinaryKernel<D:Dtype + Copy=f32> : Clone + Copy + Send + Sync + 'static {
    fn f(&self, x: D, y: D) -> D;

    fn df_dx(&self, x: D, y: D) -> D;
    fn df_dy(&self, x: D, y: D) -> D;

    fn with_left_scalar<Op: UnaryKernel<D>>(&self, _lhs: D) -> Option<Op> {
        None
    }

    fn with_right_scalar<Op: UnaryKernel<D>>(&self, _rhs: D) -> Option<Op> {
        None
    }
}

#[derive(Debug, Clone)]
pub struct BinopImpl<Op: BinaryKernel> {
    op: Op,
    size: usize,
    batch: usize,
    inner: usize,
    shape: Shape,
}

#[derive(Debug, Clone)]
pub struct BinopDx<Op: BinaryKernel>(Op);

#[derive(Debug, Clone)]
pub struct BinopDy<Op: BinaryKernel>(Op);

pub fn binary_op<Op: BinaryKernel<f32>>(a: &Tensor<f32>, b: &Tensor<f32>, op: Op) -> Tensor {
    let size = a.broadcast(b);
    let inner = a.len().min(b.len());
    let batch = size / inner;

    let shape = if a.rank() < b.rank() { 
        b.shape().clone() 
    } else { 
        a.shape().clone() 
    };

    let binop = BinopImpl {
        op: op.clone(),
        size,
        batch,
        inner,
        shape,
    };

    let node = NodeOp::new(&[a, b], binop.to_op());

    let tensor = binop.f(&[a, b], node);

    Tape::set_tensor(tensor)
}

impl<Op: BinaryKernel<f32>> Operation for BinopImpl<Op> {
    fn name(&self) -> &str {
        type_name::<Op>()
    }
    
    fn f(
        &self,
        args: &[&Tensor],
        id: TensorId,
    ) -> Tensor {
        let a = args[0];
        let b = args[1];

        let op = self.op;

        let size = self.size;
        let inner = self.inner;
        let batch = self.batch;
        
        unsafe {
            let mut out = TensorUninit::<f32>::new(size);

            for n in 0..batch {
                let a = a.as_wrap_slice(n * inner);
                let b = b.as_wrap_slice(n * inner);
                let o = out.as_sub_slice(n * inner, inner);

                for k in 0..inner {
                    o[k] = op.f(a[k], b[k]);
                }
            }

            Tensor::from_uninit_with_id(out, self.shape.clone(), id)
        }
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
            0 => graph.add_grad_op(BinopDx(self.op.clone()), &[args[0], args[1]], prev),
            1 => graph.add_grad_op(BinopDy(self.op.clone()), &[args[0], args[1]], prev),
            _ => unimplemented!(),
        }
    }
}

impl<Op:BinaryKernel<f32>> GradientOp for BinopDx<Op> {
    fn name(&self) -> &str {
        type_name::<Op>()
    }

    fn df(
        &self,
        args: &[&Tensor],
        prev: &Tensor,
    ) -> Tensor {
        let x = args[0];
        let y = args[1];
        let o_len = x.len();
        let y_len = y.len();

        let max_len = cmp::max(y_len, o_len);
        assert_eq!(max_len, prev.len());

        let batch = max_len / o_len;
        
        unsafe {
            let mut out = TensorUninit::<f32>::new(o_len);

            let x_ptr = x.as_ptr();
            let prev = prev.as_ptr();

            let o_ptr = out.as_mut_ptr();
    
            let op = &self.0;

            for i in 0..o_len {
                let mut dl_dx = 0.0f32;

                let x = *x_ptr.add(i);

                for n in 0..batch {
                    let y = *y.as_wrap_ptr(i + n * o_len);

                    let dl_df = *prev.add(i + n * o_len);

                    dl_dx += op.df_dx(x, y) * dl_df;
                }

                *o_ptr.add(i) = dl_dx;
            }
    
            Tensor::from_uninit(out, x.shape())
        }
    }
}

impl<Op:BinaryKernel<f32>> GradientOp for BinopDy<Op> {
    fn name(&self) -> &str {
        type_name::<Op>()
    }

    fn df(
        &self,
        args: &[&Tensor],
        prev: &Tensor,
    ) -> Tensor {
        let x = args[0];
        let y = args[1];
        let x_len = x.len();
        let o_len = y.len();

        let max_len = cmp::max(x_len, o_len);
        let batch = max_len / o_len;
        
        unsafe {
            let mut out = TensorUninit::<f32>::new(o_len);

            let y_ptr = y.as_ptr();
            let prev = prev.as_ptr();

            let o_ptr = out.as_mut_ptr();
    
            let op = &self.0;

            for i in 0..o_len {
                let mut dl_dy = 0.0f32;

                let y = *y_ptr.add(i);

                for n in 0..batch {
                    let x = *x.as_wrap_ptr(i + n * o_len);

                    let dl_df = *prev.add(i + n * o_len);

                    dl_dy += op.df_dy(x, y) * dl_df;
                }

                *o_ptr.add(i) = dl_dy;
            }
    
            Tensor::from_uninit(out, y.shape())
        }
    }
}

// TODO: debug seems wrong
impl<F, D:Dtype + Copy> BinaryKernel<D> for F
where F: Fn(D, D) -> D + Send + Sync + 'static + Clone + Copy {
    fn f(&self, x: D, y: D) -> D {
        (self)(x, y)
    }

    fn df_dx(&self, _x: D, _y: D) -> D {
        todo!()
    }

    fn df_dy(&self, _x: D, _y: D) -> D {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use crate::{prelude::{*}, ops::binary_op, model::Var};

    #[test]
    fn binop_broadcast() {
        let a = tf32!([1., 2., 3.]);
        let b = tf32!(1.);

        assert_eq!(
            binary_op(&a, &b, |a, b| 100. * a + b),
            tf32!([101., 201., 301.])
        );

        assert_eq!(
            binary_op(&b, &a, |a, b| 100. * a + b),
            tf32!([101., 102., 103.])
        );

        let a = tf32!([1., 2.]);
        let b = tf32!([[1., 2.], [3., 4.]]);

        assert_eq!(
            binary_op(&a, &b, |a, b| 100. * a + b),
            tf32!([[101., 202.], [103., 204.]])
        );

        assert_eq!(
            binary_op(&b, &a, |a, b| 100. * a + b),
            tf32!([[101., 202.], [301., 402.]])
        );

        let a = tf32!([1., 2.]);
        let b = tf32!([
            [[1., 2.], [3., 4.]],
            [[10., 20.], [30., 40.]],
        ]);

        assert_eq!(
            binary_op(&a, &b, |a, b| 100. * a + b),
            tf32!([
                [[101., 202.], [103., 204.]],
                [[110., 220.], [130., 240.]],
            ])
        );

        assert_eq!(
            binary_op(&b, &a, |a, b| 100. * a + b),
            tf32!([
                [[101., 202.], [301., 402.]],
                [[1001., 2002.], [3001., 4002.]],
            ]),
        );
    }

    #[test]
    fn binop_df() {
        let x = Var::new("x", tensor!([1., 2.]));
        let y = Var::new("y", tensor!([3., 4.]));

        let module = Trainer::compile((), |()| {
            &x + &y
        });
        let train = module.train(());

        assert_eq!(train.value(), tensor!([4., 6.]));
        assert_eq!(train.gradient(&x), tensor!([1., 1.]));
        assert_eq!(train.gradient(&y), tensor!([1., 1.]));
    }

    #[test]
    fn binop_df_dx_batch() {
        let x = Var::new("x", tensor!([3.]));
        let y = tf32!([[1.], [2.]]);

        let module = Trainer::compile((), |()| {
            &x + &y
        });
        let train = module.train(());

        assert_eq!(train.value(), tensor!([[4.], [5.]]));
        assert_eq!(train.gradient(&x), tensor!([2.]));
    }

    #[test]
    fn binop_df_dy_batch() {
        let x = tf32!([[1.], [2.]]);
        let y = Var::new("y", tensor!([3.]));

        let module = Trainer::compile((), |()| {
            &x + &y
        });
        let train = module.train(());

        assert_eq!(train.value(), tensor!([[4.], [5.]]));
        assert_eq!(train.gradient(&y), tensor!([2.]));
    }
}
