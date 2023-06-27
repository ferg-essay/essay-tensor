use std::any::type_name;

use crate::{
    Tensor, 
    tensor::{Dtype, TensorUninit, TensorId}, 
    model::{NodeOp, Tape, Operation, IntoForward, Expr, expr::GradientOp}
};

pub trait Softmax<D:Dtype=f32> : Clone + Copy + Send + Sync + 'static {
    fn weight(&self, a: D) -> D;

    fn f(&self, a: D, weight: D, inv_sum: D) -> D;

    fn df_dx(&self, a: D) -> D;
}

#[derive(Clone, Copy, PartialEq)]
pub struct SoftmaxCpu<Op:Softmax>(Op, usize);

#[derive(Clone, Copy, PartialEq)]
pub struct SoftmaxImpl;

pub fn softmax(a: &Tensor) -> Tensor {
    softmax_op(a, SoftmaxImpl, None)
}

impl Tensor {
    pub fn softmax(&self) -> Tensor {
        softmax(self)
    }
}

impl Softmax for SoftmaxImpl {
    fn weight(&self, a: f32) -> f32 {
        a.exp()
    }

    fn f(&self, _a: f32, weight: f32, inv_sum: f32) -> f32 {
        weight * inv_sum
    }

    fn df_dx(&self, a: f32) -> f32 {
        a.exp()
    }
}

pub fn softmax_op<Op>(a: &Tensor, op: Op, chunk: Option<usize>) -> Tensor
where
    Op:Softmax
{
    let chunk = match chunk {
        Some(chunk) => chunk,
        None => a.dim_tail(),
    };

    assert!(a.dim_tail() % chunk == 0);

    let softmax_op = SoftmaxCpu(op.clone(), chunk);

    let node = NodeOp::new(&[a], softmax_op.to_op());

    let tensor = softmax_op.f(&[&a], node);

    Tape::set_tensor(tensor)
}

impl<Op:Softmax> SoftmaxCpu<Op> {
    #[inline]
    fn op(&self) -> Op {
        self.0
    }
}

impl<Op:Softmax> Operation for SoftmaxCpu<Op> {
    fn name(&self) -> &str {
        type_name::<Op>()
    }

    fn f(
        &self,
        args: &[&Tensor],
        id: TensorId,
    ) -> Tensor {
        assert!(args.len() == 1);

        let a = args[0];

        let shape = a.shape();

        let len = a.len();
        let inner_len = a.dim_tail();
        let batch = len / inner_len;

        let chunk = self.1;
        let k_chunks = inner_len / chunk;
    
        unsafe {
            let mut o_data = TensorUninit::<f32>::new(len);

            let op = self.op();
    
            for batch in 0..batch {
                for k in 0..k_chunks {
                    let a_ptr = a.as_ptr().add(batch * inner_len + k * chunk);
                    let o_ptr = o_data.as_mut_ptr().add(batch * inner_len + k * chunk);
        
                    let mut sum = 0.;

                    for i in 0..chunk {
                        let v = op.weight(*a_ptr.add(i));

                        *o_ptr.add(i) = v;

                        sum += v;
                    }

                    // TODO: consider a softmax that doesn't select an option
                    // when all the inputs are too low
                    let factor = if sum <= 1e-20 { 1. } else { sum.recip() };

                    // normalize
                    for i in 0..chunk {
                        let v = op.f(
                            *a_ptr.add(i),
                            *o_ptr.add(i), 
                            factor
                        );

                        *o_ptr.add(i) = v;
                    }
                }
            }

            Tensor::from_uninit_with_id(o_data, shape, id)
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
        assert!(i == 0);

        graph.add_grad_op(self.clone(), &[args[0]], prev)
    }
}

impl<Op:Softmax> GradientOp for SoftmaxCpu<Op> {
    fn name(&self) -> &str {
        type_name::<Op>()
    }

    fn df(
        &self,
        args: &[&Tensor],
        prev: &Tensor,
    ) -> Tensor {
        let a = &args[0];

        assert_eq!(a.len(), prev.len());
        
        let len = a.len();
        
        unsafe {
            let mut out = TensorUninit::<f32>::new(len);

            let op = &self.0;

            let a_ptr = a.as_slice();
            let prev_ptr = prev.as_slice();
        
            for i in 0..len {
                let df_dx = op.df_dx(a_ptr[i]);
                let prev_df = prev_ptr[i];

                out[i] = df_dx * prev_df;
            }
    
            Tensor::from_uninit(out, a.shape())
        }
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_softmax() {
        assert_eq!(tf32!(0.).softmax(), tf32!(1.));
        assert_eq!(tf32!(1.).softmax(), tf32!(1.));
        assert_eq!(
            tf32!([0., 1.]).softmax(), 
            tf32!([0.26894143, 0.7310586]));
        assert_eq!(
            tf32!([0., 1., 0., 0.]).softmax(), 
            tf32!([0.1748777, 0.47536686, 0.1748777, 0.1748777]));
        assert_eq!(
            tf32!([0., 10., 0., 0.]).softmax(), 
            tf32!([4.539375e-5, 0.99986386, 4.539375e-5, 4.539375e-5]));
    }
}