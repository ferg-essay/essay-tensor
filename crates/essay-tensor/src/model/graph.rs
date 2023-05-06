use core::fmt;
use std::{collections::{HashMap, HashSet}, ops::{self}, any::type_name};

use crate::{Tensor, model::Tape, tensor::{Dtype, NodeId}};

use super::{TensorId, Var};

pub struct Graph {
    var_map: HashMap<String, TensorId>,

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
    BackOp(TensorId, BoxForwardOp, Vec<TensorId>),
}

pub trait ForwardOp : Send + Sync + 'static {
    fn eval(
        &self,
        tensors: &TensorCache,
        args: &[&Tensor],
    ) -> Tensor;

    fn backprop(
        &self,
        forward: &Graph,
        graph: &mut Graph,
        i: usize,
        args: &[TensorId],
        out: TensorId,
        prev: TensorId,
    ) -> TensorId;

    fn backprop_top(
        &self,
        forward: &Graph,
        graph: &mut Graph,
        i: usize,
        args: &[TensorId],
        out: TensorId,
    ) -> TensorId;

    fn box_clone(&self) -> BoxForwardOp;
}

pub trait EvalOp : Send + Sync + 'static {
    fn eval(
        &self,
        tensors: &TensorCache,
        args: &[&Tensor],
    ) -> Tensor;
}

impl<Op:EvalOp> ForwardOp for Op {
    fn eval(
        &self,
        tensors: &TensorCache,
        args: &[&Tensor],
    ) -> Tensor {
        (self as &dyn EvalOp).eval(tensors, args)
    }

    fn backprop(
        &self,
        _forward: &Graph,
        _graph: &mut Graph,
        _i: usize,
        _args: &[TensorId],
        _tensor: TensorId,
        _prev: TensorId,
    ) -> TensorId {
        panic!("{} does not implement backprop", type_name::<Op>())
    }

    fn backprop_top(
        &self,
        _forward: &Graph,
        _graph: &mut Graph,
        _i: usize,
        _args: &[TensorId],
        _tensor: TensorId,
    ) -> TensorId {
        panic!("{} does not implement backprop", type_name::<Op>())
    }

    fn box_clone(&self) -> BoxForwardOp {
        panic!("{} does not implement box_clone", type_name::<Op>())
    }
}

pub type BoxForwardOp = Box<dyn ForwardOp>;
/*
pub trait BackOp : fmt::Debug + Send + Sync + 'static {
    fn gradient(
        &self, 
        i: usize, 
        args: &[&Tensor],
        prev: &Option<Tensor>, 
    ) -> Tensor {
        todo!("{:?}", self)
    }

    //fn box_clone(&self) -> BoxForwardOp;
}

pub type BoxBackOp = Box<dyn BackOp>;
*/

#[derive(Clone, Debug)]
pub struct ConstOp<D:Dtype>(Tensor<D>);

pub struct ArgTrace {
    arg_index: usize,
    backtrace: BackTrace
}

impl ArgTrace {
    fn new(index: usize, backtrace: BackTrace) -> Self {
        Self {
            arg_index: index,
            backtrace,
        }
    }

    pub(crate) fn index(&self) -> usize {
        self.arg_index
    }

    pub(crate) fn backtrace(&self) -> &BackTrace {
        &self.backtrace
    }
}

pub struct BackTrace {
    pub id: TensorId,
    pub args: Vec<ArgTrace>,
}

impl<D:Dtype> ForwardOp for ConstOp<D> {
    fn box_clone(&self) -> BoxForwardOp {
        Box::new(self.clone())
    }

    fn backprop(
        &self,
        forward: &Graph,
        graph: &mut Graph,
        i: usize,
        args: &[TensorId],
        tensor: TensorId,
        prev: TensorId,
    ) -> TensorId {
        todo!()
    }

    fn backprop_top(
        &self,
        forward: &Graph,
        graph: &mut Graph,
        i: usize,
        args: &[TensorId],
        tensor: TensorId,
    ) -> TensorId {
        todo!()
    }

    fn eval(
        &self,
        tensors: &TensorCache,
        args: &[&Tensor],
    ) -> Tensor {
        todo!()
    }
}

pub trait IntoForward {
    fn to_op(&self) -> BoxForwardOp;
}

impl IntoForward for BoxForwardOp {
    fn to_op(&self) -> BoxForwardOp {
        self.box_clone()
    }
}

impl<Op> IntoForward for Op
where
    Op: Clone + ForwardOp
{
    fn to_op(&self) -> BoxForwardOp {
        Box::new(self.clone())
    }
}

impl Graph {
    pub(crate) fn var(&mut self, name: &str) -> TensorId {
        let len = self.nodes.len();
        let id = *self.var_map
            .entry(name.to_string())
            .or_insert(TensorId(len));

        if id.index() == len {
            self.nodes.push(NodeOp::Var(id, name.to_string()));
            self.tensors.push(None);
        }

        id
    }

    pub(crate) fn get_var(&self, var: &Var) -> TensorId {
       *self.var_map.get(var.name()).unwrap()
    }

    pub(crate) fn constant(&mut self, tensor: Tensor) -> TensorId {
        let id = self.alloc_id();

        self.set_tensor(id, tensor.clone());
        self.set_node(id, NodeOp::Const(id));

        id
    }

    pub(crate) fn constant_id(&mut self, forward_id: TensorId) -> TensorId {
        let back_id = self.alloc_id();

        self.set_node(back_id, NodeOp::BackConst(back_id, forward_id));

        back_id
    }

    pub(crate) fn add_op(
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
        into_op: impl IntoForward,
        prev: &[TensorId]
    ) -> TensorId {
        let id = self.alloc_id();
        self.set_node(id, NodeOp::BackOp(id, into_op.to_op(), prev.into()));
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
        self.nodes[id.index()] = node;
    }

    pub(crate) fn backtrace_graph(&mut self, target: TensorId) -> Graph {
        let backtrace = self.build_backtrace(target).unwrap();

        let back_cache = TensorCache::default();

        let mut graph = Graph::default();

        self.backtrace_graph_rec(&mut graph, &backtrace, None);

        assert!(graph.len() > 0, "backtrace produced empty graph");

        graph
    }

    pub(crate) fn backtrace_graph_rec(
        &mut self,
        graph: &mut Graph,
        backtrace: &BackTrace,
        prev: Option<TensorId>,
    ) {
        //let args : Vec<TensorId> = backtrace.args.iter().map(|arg| {
            
        //}).collect();

        let len = backtrace.args.len();

        if len == 0 {
        //let node = self.node(backtrace.id);
            let back_args = self.node_backtrace(graph, 0, backtrace.id, prev);
        }

        for arg in &backtrace.args {
            let back_arg = self.node_backtrace(graph, arg.index(), backtrace.id, prev);

            self.backtrace_graph_rec(graph, arg.backtrace(), Some(back_arg));
        }
    }

    fn node_backtrace(
        &mut self, 
        graph: &mut Graph,
        i: usize,
        id: TensorId,
        prev: Option<TensorId>
    ) -> TensorId {
        match self.node(id) {
            NodeOp::None => todo!(),
            NodeOp::Const(id) => todo!(),
            NodeOp::Var(id, name) => {
                match prev {
                    Some(prev) => {
                        prev
                    },
                    None => {
                        let id = graph.constant(Tensor::ones(self.tensor(*id).unwrap().shape()));

                        id
                    }
                }
            }
            NodeOp::Op(id, op, args) => {
                match prev {
                    Some(prev) => {
                        op.backprop(self, graph, i, args, *id, prev)
                    },
                    None => {
                        op.backprop_top(self, graph, i, args, *id)
                    }
                }
            },
            NodeOp::BackConst(_, _) => panic!("BackConst is invalid when generating backtrace"),
            NodeOp::BackOp(_, _, _) => panic!("BackConst is invalid when generating backtrace"),
        }
    }

    pub(crate) fn build_backtrace(&self, target: TensorId) -> Option<BackTrace> {
        let tail_id = TensorId(self.nodes.len() - 1);

        self.build_backtrace_rec(target, tail_id, &mut HashSet::new())
    }

    fn build_backtrace_rec(
        &self, 
        target: TensorId,
        id: TensorId,
        visited: &mut HashSet<TensorId>,
    ) -> Option<BackTrace> {
        if target == id {
            Some(BackTrace {
                id,
                args: Default::default(),
            })
        } else if visited.contains(&id) {
            None
        } else {
            match &self.nodes[id.index()] {
                NodeOp::None => None,
                NodeOp::Const(_) => None,
                NodeOp::Var(_, _) => None,
                NodeOp::Op(_, _, args) => {
                    visited.insert(id);

                    let mut next_args = Vec::<ArgTrace>::new();

                    for (i, arg) in args.iter().enumerate() {
                        if let Some(trace) =  self.build_backtrace_rec(target, *arg, visited) {
                            next_args.push(ArgTrace::new(i, trace));
                        }
                    }

                    if next_args.len() > 0 {
                        Some(BackTrace {
                            id,
                            args: next_args
                        })
                    } else {
                        None
                    }

                },
                NodeOp::BackConst(_, _) => {
                    panic!("BackConst is invalid in this context");
                },
                NodeOp::BackOp(_, _, _) => {
                    panic!("BackOp is invalid in this context");
                },
            }
        }
    }

    pub fn tensor(&self, id: TensorId) -> Option<&Tensor> {
        self.tensors.get(id)
    }

    pub(crate) fn set_tensor(&mut self, id: TensorId, tensor: Tensor) {
        self.tensors.set(id, tensor);
    }

    pub(crate) fn apply(
        &self, 
        fwd_tensors: &TensorCache, 
        args: &[&Tensor]
    ) -> TensorCache {
        assert!(self.nodes.len() > 0);

        let mut tensors_out = self.tensors.clone();
            // TODO: fill args
        for node in self.nodes.iter() {
            let tensor = node.eval(&tensors_out, &fwd_tensors);

            // TODO: 
            tensors_out.set(node.id(), tensor);
        }

        tensors_out
    }

    pub(crate) fn tensors(&self) -> &TensorCache {
        &self.tensors
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self { 
            var_map: Default::default(),
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
                NodeId::Var(name) => Tape::var(name),
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
            NodeOp::Var(_, _) => todo!(),
            NodeOp::Op(id, op, args) => {
                let t_args: Vec<&Tensor> = args.iter()
                    .map(|id| tensors.get(*id).unwrap())
                    .collect();

                op.eval(tensors, &t_args)
            },

            NodeOp::BackConst(_, forward_id) => {
                fwd_tensors[*forward_id].clone()
            },
            NodeOp::BackOp(id, op, args) => {
                let t_args: Vec<&Tensor> = args.iter()
                    .map(|id| fwd_tensors.get(*id).unwrap())
                    .collect();

                op.eval(tensors, &t_args)
            },
        }
    }

    fn constant(tensor: &Tensor) -> TensorId {
        let id = Tape::alloc_id().unwrap();

        Tape::set_tensor(id, tensor.clone());
        Tape::set_node(id, NodeOp::Const(id));

        id
    }

    pub(crate) fn gradient(&self, tape: &Tape, arg: usize) -> Tensor {
        todo!()
    }

    fn id(&self) -> TensorId {
        match self {
            NodeOp::None => todo!(),
            NodeOp::Const(id) => *id,
            NodeOp::BackConst(id, _) => *id,
            NodeOp::Var(id, _) => *id,
            NodeOp::Op(id, _, _) => *id,
            NodeOp::BackOp(id, _, _) => *id,
        }
    }
}

impl Clone for NodeOp {
    fn clone(&self) -> Self {
        match self {
            NodeOp::None => NodeOp::None,
            NodeOp::Const(id) => NodeOp::Const(*id),
            NodeOp::Var(id, name) => NodeOp::Var(*id, name.clone()),
            NodeOp::Op(id, op, args) => {
                NodeOp::Op(*id, op.box_clone(), args.clone())
            }
            NodeOp::BackConst(_, _) => todo!(),
            NodeOp::BackOp(id, op, args) => {
                NodeOp::BackOp(*id, op.box_clone(), args.clone())
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

impl TensorCache {
    pub(crate) fn push(&mut self, tensor: Option<Tensor>) {
        self.tensors.push(tensor)
    }

    fn get(&self, id: TensorId) -> Option<&Tensor> {
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

