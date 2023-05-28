use core::fmt;
use std::{collections::{HashMap}, ops::{self}};

use log::debug;

use crate::{Tensor, tensor::{TensorId}};

use super::{Var, Tape, var::VarId};

pub struct Graph {
    var_map: HashMap<VarId, VarItem>,
    tracked_vars: Vec<Var>,

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
    Var(TensorId, VarId, String),
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
            Self::Var(id, var_id, name) => {
                write!(f, "Var({})[{}, {}]", id.index(), var_id.index(), name)
            },
            Self::Op(id, op, args) => {
                //f.debug_tuple("Op").field(id).field(&op.name()).field(args).finish()
                write!(f, "{}({}){:?}", 
                    id.index(), 
                    op.name().rsplit_once("").unwrap().1,
                    args,
                )
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

impl fmt::Debug for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Graph {{\n")?;

        for op in &self.nodes {
            write!(f, "  #{} {:?}\n", op.id().index(), op)?;
        }

        write!(f, "}}")
    }
}

pub trait Operation : Send + Sync + 'static {
    fn name(&self) -> &str;

    fn forward(
        &self,
        args: &[&Tensor],
        id: TensorId,
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

pub trait GradientOp : Send + Sync + 'static {
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

pub type BoxBackOp = Box<dyn GradientOp>;

impl Graph {
    pub(crate) fn var(&mut self, var: &Var) -> Tensor {
        let len = self.nodes.len();

        let item = self.var_map
            .entry(var.id())
            .or_insert_with(|| VarItem::new(TensorId(len), var));

        if item.id().index() == len {
            let op = NodeOp::Var(item.id(), var.id(), var.name().to_string());
            debug!("GraphVar {:?}", op);
            self.nodes.push(op);
            self.tensors.push(None);
            self.tracked_vars.push(var.clone());
        }

        var.tensor_with_id(item.id())
    }

    pub(crate) fn get_id_by_var(&self, id: VarId) -> TensorId {
        self.var_map.get(&id).unwrap().id()
    }

    pub(crate) fn get_tensor_by_var(&self, id: VarId) -> Tensor {
        let var_item = self.var_map.get(&id).unwrap();

        var_item.tensor()
    }

    pub(crate) fn tracked_vars(&self) -> &Vec<Var> {
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
    pub fn new(args: &[&Tensor], op: BoxForwardOp) -> TensorId {
        if ! Tape::is_active() {
            return TensorId::None;
        }

        let node_args : Vec<TensorId> = args.iter().map(|tensor| 
            if tensor.id().is_none() {
                Self::constant(tensor)
            } else {
                tensor.id()
            }
        ).collect();

        let id = Tape::alloc_id();

        if id.is_none() {
            return TensorId::None;
        }

        let id = id.unwrap();

        let node = NodeOp::Op(id, op, node_args);

        Tape::set_node(id, node);

        id
    }

    fn eval(
        &self, 
        tensors: &TensorCache,
        fwd_tensors: &TensorCache,
    ) -> Tensor {
        let value = match self {
            NodeOp::None => todo!(),
            NodeOp::Const(id) => tensors[*id].clone(),
            NodeOp::Var(id, var_id, _) => tensors[*id].clone(),
            NodeOp::Op(id, op, args) => {
                let t_args: Vec<&Tensor> = args.iter()
                    .map(|id| tensors.get(*id).unwrap())
                    .collect();

                op.forward(&t_args, *id)
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
        };

        println!("Eval {:?}", value);

        value
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
            NodeOp::Var(id, _, _) => *id,
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

struct VarItem {
    id: TensorId,
    var: Var,
}

impl VarItem {
    fn tensor(&self) -> Tensor {
        self.var.tensor_with_id(self.id)
    }

    fn new(id: TensorId, var: &Var) -> VarItem {
        Self {
            id,
            var: var.clone(),
        }
    }

    #[inline]
    fn id(&self) -> TensorId {
        self.id
    }

    #[inline]
    fn var(&self) -> &Var {
        &self.var
    }
}
