use core::fmt;

use crate::{Tensor, model::Tape, tensor::{BoxOp, Dtype, Op}};

use super::TensorId;

pub struct OpGraph {
    tensor_id: TensorId,
    op: BoxOp,
    args: Vec<OpGraph>,
    //tensor: Option<Tensor<D>>,
}

impl fmt::Debug for OpGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.args.len() > 0 {
            f.debug_struct("OpGraph")
                .field("op", &self.op)
                .field("args", &self.args)
                .finish()
        } else {
            self.op.fmt(f)
        }
    }
}

#[derive(Clone, Debug)]
pub struct ConstOp<D:Dtype>(Tensor<D>);

impl<D:Dtype> Op for ConstOp<D> {
    fn box_clone(&self) -> BoxOp {
        Box::new(self.clone())
    }
}

impl OpGraph {
    pub fn new<D:Dtype>(args: &[&Tensor<D>], op: BoxOp) -> OpGraph {
        Self {
            tensor_id: TensorId(0),
            args: args.iter().map(|tensor| 
                match tensor.op() {
                    Some(op) => op.clone(),
                    None => Self::constant(tensor),
                }
            ).collect(),
            op,
        }
    }

    pub fn constant<D:Dtype>(tensor: &Tensor<D>) -> OpGraph {
        Self {
            tensor_id: TensorId(0),
            args: vec![],
            op: ConstOp(tensor.clone()).box_clone(),
        }
    }

    pub(crate) fn gradient(&self, tape: &Tape, arg: usize) -> Tensor {
        todo!()
    }
}

impl Clone for OpGraph {
    fn clone(&self) -> Self {
        Self { 
            tensor_id: self.tensor_id,
            args: self.args.clone(), 
            op: self.op.box_clone()
         }
    }
}
