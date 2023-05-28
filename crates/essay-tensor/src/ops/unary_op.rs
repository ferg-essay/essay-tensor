use core::fmt;
use std::{any::type_name};

use crate::{function::{IntoForward, NodeOp, Tape, Operation, Graph, graph::GradientOp}, Tensor, 
    tensor::{Dtype, TensorUninit, TensorId}
};

pub trait UnaryKernel<D:Dtype> : fmt::Debug + Copy + Clone + PartialEq + Sync + Send + 'static
{
    fn f(&self, value: D) -> D;

    fn df_dx(&self, value: D) -> D;
}

#[derive(Clone, PartialEq)]
pub struct UopCpu<Op:UnaryKernel<f32>>(Op);

pub fn unary_op<Op>(a: &Tensor, op: Op) -> Tensor
where
    Op:UnaryKernel<f32>
{
    let uop = UopCpu(op.clone());

    let node = NodeOp::new(&[a], uop.to_op());

    let tensor = uop.forward(&[&a], node);

    Tape::set_tensor(tensor)
}

impl<Op:UnaryKernel<f32>> Operation for UopCpu<Op> {
    fn name(&self) -> &str {
        type_name::<Op>()
    }
    
    fn forward(
        &self,
        args: &[&Tensor],
        node: TensorId,
    ) -> Tensor {
        let a = args[0];
        let len = a.len();
    
        unsafe {
            let mut out = TensorUninit::<f32>::new(len);
    
            let op = &self.0;
            let a_ptr = a.as_ptr();
            let o_ptr = out.as_mut_ptr();
        
            for i in 0..len {
                *o_ptr.add(i) = op.f(*a_ptr.add(i));
            }
    
            Tensor::from_uninit_with_id(out, a.shape(), node)
        }
    }

    fn back(
        &self,
        _forward: &Graph,
        graph: &mut Graph,
        i: usize,
        args: &[TensorId],
        prev: TensorId,
    ) -> TensorId {
        assert!(i == 0);

        graph.add_grad_op(self.clone(), &[args[0]], prev)
    }
}

impl<Op:UnaryKernel<f32>> GradientOp for UopCpu<Op> {
    fn name(&self) -> &str {
        type_name::<Op>()
    }

    fn df(
        &self,
        args: &[&Tensor],
        prev: &Tensor,
    ) -> Tensor {
        let tensor = &args[0];
        let len = tensor.len();
        
        unsafe {
            let mut out = TensorUninit::<f32>::new(len);

            let ptr = tensor.as_ptr();
            let prev = prev.as_ptr();
            let o_ptr = out.as_mut_ptr();
    
            let op = &self.0;
        
            for i in 0..len {
                let df_dx = op.df_dx(*ptr.add(i));
                let prev_df = *prev.add(i);

                *o_ptr.add(i) = df_dx * prev_df;
            }
    
            Tensor::from_uninit(out, tensor.shape())
        }
    }
}