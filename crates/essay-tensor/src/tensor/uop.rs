use core::fmt;
use std::sync::Arc;

use crate::{module::{IntoForward, NodeOp, Tape, ModuleTape}, Tensor};

use super::{Dtype, TensorUninit, TensorData, NodeId};


pub trait Uop<D:Dtype> : fmt::Debug + Clone + Sync + Send + 'static {
    fn eval(&self, value: D) -> D;
}

impl Tensor {
    pub fn uop<Op>(&self, uop: Op) -> Self
    where
        Op:Uop<f32> + IntoForward
    {
        let buffer = self.data();
        let len = buffer.len();

        unsafe {
            let mut data = TensorUninit::<f32>::new(len);
    
            for i in 0..len {
                data.set_unchecked(i, uop.eval(buffer.get_unchecked(i)));
            }
    
            let shape = self.shape().clone();
            self.next_uop(data.init(), Vec::from(shape), uop)
        }
    }

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
            // Tape::set_tensor(*id, tensor.clone());
            ModuleTape::set_tensor(*id, tensor.clone());
        }

        tensor
    }
}

 
impl<F, D:Dtype> Uop<D> for F
where F: Fn(D) -> D + Clone + fmt::Debug + Sync + Send + 'static {
    fn eval(&self, value: D) -> D {
        (self)(value)
    }
}
