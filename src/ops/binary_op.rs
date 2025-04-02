use std::{any::type_name, cmp, marker::PhantomData};

use crate::{model::{Expr, Operation, expr::{GradientOp, GradOperation}}, Tensor, 
    tensor::{Dtype, TensorId, TensorUninit},
};

use super::UnaryKernel;

pub trait BinaryKernel<D: Dtype=f32> : Clone + Send + Sync + 'static {
    fn f(&self, x: &D, y: &D) -> D;

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
pub struct BinopImpl<D: Dtype, Op: BinaryKernel<D>> {
    op: Op,
    marker: PhantomData<D>,
}

#[derive(Debug, Clone)]
pub struct BinopDx<Op: BinaryKernel>(Op);

#[derive(Debug, Clone)]
pub struct BinopDy<Op: BinaryKernel>(Op);

pub fn binary_op<D, Op>(
    a: impl Into<Tensor<D>>, 
    b: impl Into<Tensor<D>>, 
    op: Op
) -> Tensor<D>
where
    D: Dtype + Clone,
    Op: BinaryKernel<D>,
{
    let a = a.into();
    let b = b.into();

    let binop = BinopImpl {
        op: op.clone(),
        marker: PhantomData,
    };

    let id = D::node_op(&[&a, &b], &binop); // .to_op());

    let tensor = binop.f(&[&a, &b], id);

    D::set_tape(tensor)
}

impl<D, Op> Operation<D> for BinopImpl<D, Op>
where
    D: Dtype,
    Op: BinaryKernel<D>,
{
    fn name(&self) -> &str {
        type_name::<Op>()
    }
    
    fn f(
        &self,
        args: &[&Tensor<D>],
        id: TensorId,
    ) -> Tensor<D> {
        let a = args[0];
        let b = args[1];
    
        let op = &self.op;

        let size = a.broadcast(&b);
        let inner = a.len().min(b.len());
        let batch = size / inner;
        
        unsafe {
            let mut out = TensorUninit::<D>::new(size);

            for n in 0..batch {
                let a = a.as_wrap_slice(n * inner);
                let b = b.as_wrap_slice(n * inner);
                let o = out.as_sub_slice(n * inner, inner);

                for k in 0..inner {
                    o[k] = op.f(&a[k], &b[k]);
                }
            }

            let shape = if a.rank() < b.rank() { 
                b.shape()
            } else { 
                a.shape()
            };
    
            out.into_tensor_with_id(shape, id)
        }
    }
}


impl<D, Op> GradOperation<D> for BinopImpl<D, Op>
where
    D: Dtype,
    Op: BinaryKernel<D>,
{
    fn df(
        &self,
        _forward: &Expr,
        _graph: &mut Expr,
        i: usize,
        _args: &[TensorId],
        _prev: TensorId,
    ) -> TensorId {
        match i {
            0 => todo!(), // graph.add_grad_op(BinopDx(self.op.clone()), &[args[0], args[1]], prev),
            1 => todo!(), // graph.add_grad_op(BinopDy(self.op.clone()), &[args[0], args[1]], prev),
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

#[cfg(test)]
mod test {
    use crate::{prelude::{*}, model::Var};

    #[test]
    fn binop_df() {
        let x = Var::new("x", tensor!([1., 2.]));
        let y = Var::new("y", tensor!([3., 4.]));

        let module = Trainer::compile((), |(), _| {
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

        let module = Trainer::compile((), |(), _| {
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

        let module = Trainer::compile((), |(), _| {
            &x + &y
        });
        let train = module.train(());

        assert_eq!(train.value(), tensor!([[4.], [5.]]));
        assert_eq!(train.gradient(&y), tensor!([2.]));
    }
}
