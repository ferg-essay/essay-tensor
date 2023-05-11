use core::fmt;
use std::{collections::{HashMap}, ops::{self}};

use log::debug;

use crate::{Tensor, tensor::{NodeId}};

use super::{Var, Tape};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(pub usize);

pub struct Graph {
    var_map: HashMap<String, TensorId>,
    tracked_vars: Vec<String>,

    nodes: Vec<NodeOp>,
    tensors: TensorCache,
}

#[derive(Clone)]
pub struct TensorCache {
    tensors: Vec<Option<Tensor>>,
}

pub enum NodeOp {
    None,
    Const(TensorId),
    Var(TensorId, String),
    Op(TensorId, BoxForwardOp, Vec<TensorId>),

    BackConst(TensorId, TensorId),
    BackOp(TensorId, BoxBackOp, Vec<TensorId>, TensorId),
}

impl fmt::Debug for NodeOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::Const(arg0) => {
                f.debug_tuple("Const").field(arg0).finish()
            },
            Self::Var(arg0, arg1) => {
                f.debug_tuple("Var").field(arg0).field(arg1).finish()
            },
            Self::Op(id, op, args) => {
                f.debug_tuple("Op").field(id).field(&op.name().to_string()).field(args).finish()
            },
            Self::BackConst(arg0, arg1) => {
                f.debug_tuple("BackConst").field(arg0).field(arg1).finish()
            },
            Self::BackOp(arg0, op, arg2, arg3) => {
                f.debug_tuple("BackOp").field(arg0).field(&op.name().to_string()).field(arg2).field(arg3).finish()
            },
        }
    }
}

pub trait Operation : Send + Sync + 'static {
    fn name(&self) -> &str;

    fn forward(
        &self,
        args: &[&Tensor],
        node: NodeId,
    ) -> Tensor;

    fn back(
        &self,
        forward: &Graph,
        back: &mut Graph,
        i: usize,
        args: &[TensorId],
        prev: TensorId,
    ) -> TensorId;
}

pub trait IntoForward {
    fn to_op(&self) -> BoxForwardOp;
}

pub trait EvalOp : Send + Sync + 'static {
    fn eval(
        &self,
        args: &[&Tensor],
    ) -> Tensor;
}

pub type BoxForwardOp = Box<dyn Operation>;

pub trait BackOp : Send + Sync + 'static {
    fn name(&self) -> &str;

    fn df(
        &self,
        args: &[&Tensor],
        prev: &Tensor,
    ) -> Tensor;
}

pub trait IntoBack {
    fn to_op(&self) -> BoxBackOp;
}

pub type BoxBackOp = Box<dyn BackOp>;

impl Graph {
    pub(crate) fn var(&mut self, name: &str, tensor: &Tensor) -> TensorId {
        let len = self.nodes.len();
        let id = *self.var_map
            .entry(name.to_string())
            .or_insert(TensorId(len));

        if id.index() == len {
            let op = NodeOp::Const(id);
            debug!("GraphVar {} {:?}", name, op);
            self.nodes.push(op);
            self.tensors.push(Some(tensor.clone()));
            self.tracked_vars.push(name.to_string());
        }

        id
    }

    pub(crate) fn find_var(&self, name: &str) -> TensorId {
        *self.var_map.get(name).unwrap()
    }

    pub(crate) fn _get_var(&self, var: &Var) -> TensorId {
        *self.var_map.get(var.name()).unwrap()
    }

    pub(crate) fn tracked_vars(&self) -> &Vec<String> {
        &self.tracked_vars
    }

    pub(crate) fn constant(&mut self, tensor: Tensor) -> TensorId {
        let id = self.alloc_id();

        self.set_tensor(id, tensor.clone());
        self.set_node(id, NodeOp::Const(id));

        id
    }

    pub(crate) fn _constant_id(&mut self, forward_id: TensorId) -> TensorId {
        let back_id = self.alloc_id();

        self.set_node(back_id, NodeOp::BackConst(back_id, forward_id));

        back_id
    }

    pub(crate) fn _add_op(
        &mut self, 
        into_op: impl IntoForward,
        prev: &[TensorId]
    ) -> TensorId {
        let id = self.alloc_id();
        self.set_node(id, NodeOp::Op(id, into_op.to_op(), prev.into()));
        id
    }

    pub(crate) fn add_back_op(
        &mut self, 
        into_op: impl IntoBack,
        args: &[TensorId],
        prev: TensorId,
    ) -> TensorId {
        let id = self.alloc_id();
        self.set_node(id, NodeOp::BackOp(id, into_op.to_op(), args.into(), prev));
        id
    }

    pub(crate) fn alloc_id(&mut self) -> TensorId {
        let id = TensorId(self.nodes.len());

        self.nodes.push(NodeOp::None);
        self.tensors.push(None);

        id
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn node(&self, id: TensorId) -> &NodeOp {
        &self.nodes[id.index()]
    }

    pub(crate) fn set_node(&mut self, id: TensorId, node: NodeOp) {
        debug!("Graph.set_node [{}] {:?}", id.index(), node);
        self.nodes[id.index()] = node;
    }

    pub fn tensor(&self, id: TensorId) -> Option<&Tensor> {
        self.tensors.get(id)
    }

    pub(crate) fn set_tensor(&mut self, id: TensorId, tensor: Tensor) {
        self.tensors.set(id, tensor);
    }

    pub(crate) fn apply(
        &self, 
        out: &mut TensorCache,
        fwd_tensors: &TensorCache, 
    ) {
        assert!(self.nodes.len() > 0);

        // let mut tensors_out = self.tensors.clone();
            // TODO: fill args
        for node in self.nodes.iter() {
            let tensor = node.eval(&out, &fwd_tensors);

            // TODO: 
            out.set(node.id(), tensor);
        }
    }

    pub(crate) fn tensors(&self) -> &TensorCache {
        &self.tensors
    }

    pub(crate) fn tail_id(&self) -> TensorId {
        TensorId(self.nodes.len() - 1)
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self { 
            var_map: Default::default(),
            tracked_vars: Default::default(),

            nodes: Default::default(), 
            tensors: Default::default(), 
        }
    }
}

impl NodeOp {
    pub fn new(args: &[&Tensor], op: BoxForwardOp) -> NodeId {
        if ! Tape::is_active() {
            return NodeId::None;
        }

        let node_args : Vec<TensorId> = args.iter().map(|tensor| 
            match tensor.op() {
                NodeId::None => Self::constant(tensor),
                NodeId::Id(id) => *id,
                NodeId::Var(name) => Tape::find_var(name),
            }
        ).collect();

        let id = Tape::alloc_id();

        if id.is_none() {
            return NodeId::None;
        }

        let id = id.unwrap();

        let node = NodeOp::Op(id, op, node_args);

        Tape::set_node(id, node);

        NodeId::Id(id)
    }

    fn eval(
        &self, 
        tensors: &TensorCache,
        fwd_tensors: &TensorCache,
    ) -> Tensor {
        match self {
            NodeOp::None => todo!(),
            NodeOp::Const(id) => tensors[*id].clone(),
            NodeOp::Var(_, _) => {
                panic!()
            },
            NodeOp::Op(id, op, args) => {
                let t_args: Vec<&Tensor> = args.iter()
                    .map(|id| tensors.get(*id).unwrap())
                    .collect();

                op.forward(&t_args, NodeId::Id(*id))
            },

            NodeOp::BackConst(_, forward_id) => {
                fwd_tensors[*forward_id].clone()
            },
            NodeOp::BackOp(_, op, args, prev) => {
                let t_args: Vec<&Tensor> = args.iter()
                    .map(|id| fwd_tensors.get(*id).unwrap())
                    .collect();

                op.df(&t_args, tensors.get(*prev).unwrap())
            },
        }
    }

    fn constant(tensor: &Tensor) -> TensorId {
        let id = Tape::alloc_id().unwrap();

        Tape::set_tensor_id(id, tensor);
        Tape::set_node(id, NodeOp::Const(id));

        id
    }

    fn id(&self) -> TensorId {
        match self {
            NodeOp::None => todo!(),
            NodeOp::Const(id) => *id,
            NodeOp::BackConst(id, _) => *id,
            NodeOp::Var(id, _) => *id,
            NodeOp::Op(id, _, _) => *id,
            NodeOp::BackOp(id, _, _, _) => *id,
        }
    }
}

impl TensorCache {
    pub(crate) fn push(&mut self, tensor: Option<Tensor>) {
        self.tensors.push(tensor)
    }

    pub(crate) fn get(&self, id: TensorId) -> Option<&Tensor> {
        match &self.tensors[id.index()] {
            Some(tensor) => Some(tensor),
            None => None,
        }
    }

    pub(crate) fn set(&mut self, id: TensorId, tensor: Tensor) {
        self.tensors[id.index()] = Some(tensor);
    }

    pub(crate) fn last(&self) -> Tensor {
        match &self.tensors[self.tensors.len() - 1] {
            Some(tensor) => tensor.clone(),
            None => todo!(),
        }
    }
}

impl ops::Index<TensorId> for TensorCache {
    type Output = Tensor;

    fn index(&self, id: TensorId) -> &Self::Output {
        match &self.tensors[id.index()] {
            Some(tensor) => tensor,
            None => panic!("unset tensor {:?}", id),
        }
    }
}

impl Default for TensorCache {
    fn default() -> Self {
        Self { tensors: Default::default() }
    }
}

impl TensorId {
    #[inline]
    pub fn index(&self) -> usize {
        self.0
    }
}
