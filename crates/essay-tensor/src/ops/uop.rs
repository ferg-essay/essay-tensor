use core::fmt;
use std::sync::Arc;

use crate::{module::{IntoForward, NodeOp, Tape, EvalOp, TensorCache}, Tensor, tensor::{Dtype, TensorUninit, TensorData, NodeId}};

pub trait Uop<D:Dtype> : fmt::Debug + Clone + PartialEq + Sync + Send + 'static {
    fn f(&self, value: D) -> D;
}

#[derive(Clone)]
pub struct UopCpu<Op:Uop<f32>>(Op);

impl Tensor {
    pub fn next_uop(
        &self, 
        data: TensorData, 
        shape: Vec<usize>,
        op: impl IntoForward,
    ) -> Tensor {
        let tensor = Self::new_op(
            Arc::new(data), 
            shape, 
            NodeOp::new(&[self], op.to_op()),
        );

        if let NodeId::Id(id) = tensor.node() {
            Tape::set_tensor(*id, tensor.clone());
        }

        tensor
    }
}

pub fn uop<Op>(tensor: &Tensor, op: Op) -> Tensor
where
    Op:Uop<f32>
{
    let node = NodeOp::new(&[tensor], UopCpu(op.clone()).to_op());

    let tensor = eval_uop(tensor, &op, node);

    if let NodeId::Id(id) = tensor.node() {
        Tape::set_tensor(*id, tensor.clone());
    }

    tensor
}


impl<Op:Uop<f32>> EvalOp for UopCpu<Op> {
    fn eval(
        &self,
        _tensors: &TensorCache,
        args: &[&Tensor],
    ) -> Tensor {
        eval_uop(args[0], &self.0, NodeId::None)
    }
} 

fn eval_uop<Op:Uop<f32>>(
    tensor: &Tensor,
    op: &Op,
    node: NodeId,
) -> Tensor {
    let buffer = tensor.data();
    let len = buffer.len();
    
    let data = unsafe {
        let mut data = TensorUninit::<f32>::new(len);
    
        for i in 0..len {
            data.set_unchecked(i, op.f(buffer.get_unchecked(i)));
        }

        data.init()
    };
    
    let shape = tensor.shape().clone();
    Tensor::new_op(Arc::new(data), shape, node)
} 
