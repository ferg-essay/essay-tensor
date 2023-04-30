use core::fmt;

use crate::{Tensor, model::Tape, tensor::{BoxOp, Dtype, Op, NodeId}};

use super::TensorId;

#[derive(Debug)]
pub enum NodeOp {
    None,
    Const(TensorId),
    Var(TensorId, String),
    Op(BoxOp, Vec<TensorId>),
}

/*
pub struct NodeOp {
    tensor_id: TensorId,
    op: BoxOp,
    args: Vec<TensorId>,
    //tensor: Option<Tensor<D>>,
}

impl fmt::Debug for NodeOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.args.len() > 0 {
            f.debug_struct(&format!("NodeOp[{}]", self.tensor_id.index()))
                .field("op", &self.op)
                .field("args", &self.args.iter()
                    .map(|id| id.index())
                    .collect::<Vec<usize>>())
                .finish()
        } else {
            self.op.fmt(f)
        }
    }
}
*/

#[derive(Clone, Debug)]
pub struct ConstOp<D:Dtype>(Tensor<D>);

impl<D:Dtype> Op for ConstOp<D> {
    fn box_clone(&self) -> BoxOp {
        Box::new(self.clone())
    }
}

impl NodeOp {
    pub fn new(args: &[&Tensor], op: BoxOp) -> NodeId {
        if ! Tape::is_active() {
            return NodeId::None;
        }

        let node_args : Vec<TensorId> = args.iter().map(|tensor| 
            match tensor.op() {
                NodeId::None => Self::constant(tensor),
                NodeId::Id(id) => *id,
                NodeId::Var(name) => Tape::var(name),
            }
        ).collect();

        let id = Tape::alloc_id();

        if id.is_none() {
            return NodeId::None;
        }

        let id = id.unwrap();

        let graph = NodeOp::Op(op, node_args);

        Tape::set_graph(id, graph);

        NodeId::Id(id)
    }

    fn constant(tensor: &Tensor) -> TensorId {
        let id = Tape::alloc_id().unwrap();

        Tape::set_tensor(id, tensor.clone());
        Tape::set_graph(id, NodeOp::Const(id));

        id
    }

    pub(crate) fn gradient(&self, tape: &Tape, arg: usize) -> Tensor {
        todo!()
    }
}

impl Clone for NodeOp {
    fn clone(&self) -> Self {
        match self {
            NodeOp::None => NodeOp::None,
            NodeOp::Const(id) => NodeOp::Const(*id),
            NodeOp::Var(id, name) => NodeOp::Var(*id, name.clone()),
            NodeOp::Op(op, args) => {
                NodeOp::Op(op.box_clone(), args.clone())
            }
        }
        /*
        Self { 
            tensor_id: self.tensor_id,
            args: self.args.clone(), 
            op: self.op.box_clone()
         }
         */
    }
}

