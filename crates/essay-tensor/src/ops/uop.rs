use core::fmt;
use std::{sync::Arc, any::type_name};

use crate::{module::{IntoForward, NodeOp, Tape, EvalOp, TensorCache, ForwardOp, Graph, TensorId, graph::BackOp}, Tensor, tensor::{Dtype, TensorUninit, TensorData, NodeId}};

pub trait Uop<D:Dtype> : fmt::Debug + Clone + PartialEq + Sync + Send + 'static {
    fn f(&self, value: D) -> D;

    fn df_dx(&self, value: D) -> D;
}

#[derive(Clone)]
pub struct UopCpu<Op:Uop<f32>>(Op);

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

impl<Op:Uop<f32>> ForwardOp for UopCpu<Op> {
    fn name(&self) -> &str {
        type_name::<Op>()
    }
    
    fn eval(
        &self,
        _tensors: &TensorCache,
        args: &[&Tensor],
    ) -> Tensor {
        eval_uop(args[0], &self.0, NodeId::None)
    }

    fn backprop(
        &self,
        _forward: &Graph,
        graph: &mut Graph,
        i: usize,
        args: &[TensorId],
        prev: TensorId,
    ) -> TensorId {
        assert!(i == 0);

        graph.add_back_op(self.clone(), &[args[0]], prev)
    }
}

impl<Op:Uop<f32>> BackOp for UopCpu<Op> {
    fn name(&self) -> &str {
        type_name::<Op>()
    }

    fn df(
        &self,
        _tensors: &TensorCache,
        args: &[&Tensor],
        prev: &Tensor,
    ) -> Tensor {
        let tensor = &args[0];
        let buffer = tensor.data();
        let prev = prev.data();
        let len = buffer.len();
        
        let data = unsafe {
            let mut data = TensorUninit::<f32>::new(len);

            let op = &self.0;
        
            for i in 0..len {
                let df_dx = op.df_dx(buffer.get_unchecked(i));
                let prev_df = prev.get_unchecked(i);

                data.set_unchecked(i, df_dx * prev_df);
            }
    
            data.init()
        };
        
        let shape = tensor.shape().clone();
        Tensor::new(Arc::new(data), &shape)
    }
}