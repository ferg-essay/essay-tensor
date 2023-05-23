use std::cell::RefCell;
use crate::{Tensor, tensor::{NodeId, TensorId, Bundle}};


use super::{Var, TensorCache, Graph, NodeOp};

pub struct Tape {
    _args: Vec<TensorId>,
    _vars: Vec<(Var, TensorId)>,
    tensors: TensorCache,
    //_tail: Option<Tensor>,
    out_ids: Vec<TensorId>,

    graph: Option<Graph>,
}

#[derive(Debug)]
pub enum TapeError {}

thread_local! {
    pub static TAPE: RefCell<Option<Tape>>   = RefCell::new(None);
}

impl Tape {
    pub fn build<F, In, Out>(init: In, fun: F) -> Tape
    where
        In: Bundle<Item=In>,
        Out: Bundle<Item=Out>,
        F: FnOnce(In) -> Out,
    {
        let mut tape = Tape {
            _args: Default::default(),
            _vars: Default::default(),
            tensors: Default::default(),
            out_ids: Default::default(),

            graph: Some(Default::default()),
        };

        let len = In::push_arg(tape.tensors_mut(), 0, &init);

        // Tensors in args now have their id set.
        let mut index = 0;
        let args = In::make_arg(&tape.tensors(), &mut index);

        for id in 0..len {
            tape.constant(tape.get_tensor(TensorId(id)).unwrap().clone());
        }

        // TODO: add RAII guard?
        TAPE.with(|f| {
            assert!(f.borrow().is_none());
            f.borrow_mut().replace(tape);
        });


        let out = fun(args);
        let mut out_ids : Vec<TensorId> = Vec::new();
        Out::out_ids(&mut out_ids, &out);

        let mut tape = TAPE.with(|f| f.borrow_mut().take().unwrap());

        tape.out_ids = out_ids;

        tape
    }

    pub fn is_active() -> bool {
        TAPE.with(|f| match f.borrow().as_ref() {
            Some(_) => true,
            None => false,
        })
    }

    pub fn alloc_id() -> Option<TensorId> {
        TAPE.with(|f| match f.borrow_mut().as_mut() {
            Some(tape) => Some(tape.alloc_id_inner()),
            None => None,
        })
    }

    pub fn alloc_id_inner(&mut self) -> TensorId {
        match &mut self.graph {
            Some(graph) => graph.alloc_id(),
            None => panic!(),
        }
    }

    pub(crate) fn set_node(id: TensorId, node: NodeOp) {
        TAPE.with(|f| {
            if let Some(tape) = f.borrow_mut().as_mut() {
                tape.graph_mut().set_node(id, node);
                // tape.nodes[id.index()] = node;
            } else {
                panic!("call set_graph with no active tape");
            }
        })
    }

    pub fn get_tensor(&self, id: TensorId) -> Option<&Tensor> {
        match &self.tensors.get(id) {
            Some(tensor) => Some(tensor),
            None => None,
        }
    }

    pub(crate) fn set_tensor(tensor: Tensor) -> Tensor {
        if let NodeId::Id(id) = tensor.node_id() {
            TAPE.with(|f| {
                if let Some(tape) = f.borrow_mut().as_mut() {
                    tape.graph_mut().set_tensor(*id, tensor.clone());
                }
            })
        }

        tensor
    }

    pub(crate) fn set_tensor_id(id: TensorId, tensor: &Tensor) {
        TAPE.with(|f| {
            if let Some(tape) = f.borrow_mut().as_mut() {
                tape.graph_mut().set_tensor(id, tensor.clone());
            }
        })
    }

    pub fn var(var: &Var) -> TensorId {
        TAPE.with(|f| {
            if let Some(tape) = f.borrow_mut().as_mut() {
                tape.set_var_inner(var.name(), var.tensor_raw())
            } else {
                panic!("Tape::var without context")
            }
        })
    }

    pub fn find_var(name: &str) -> TensorId {
        TAPE.with(|f| {
            if let Some(tape) = f.borrow_mut().as_mut() {
                tape.find_var_inner(name)
            } else {
                panic!("Tape::var without context")
            }
        })
    }

    pub fn set_var(var: &str, tensor: &Tensor) -> TensorId {
        TAPE.with(|f| {
            if let Some(tape) = f.borrow_mut().as_mut() {
                tape.set_var_inner(var, tensor)
            } else {
                panic!("Tape::set_var without context")
            }
        })
    }

    pub fn find_var_inner(&mut self, new_var: &str) -> TensorId {
        self.graph().find_var(new_var)
    }

    pub fn set_var_inner(&mut self, name: &str, tensor: &Tensor) -> TensorId {
        self.graph_mut().var(name, tensor)
    }

    pub(crate) fn tracked_vars(&self) -> &Vec<String> {
        self.graph().tracked_vars()
    }

    pub(crate) fn tensors(&self) -> &TensorCache {
        &self.tensors
    }

    pub(crate) fn tensors_mut(&mut self) -> &mut TensorCache {
        &mut self.tensors
    }

    pub(crate) fn graph(&self) -> &Graph {
        match &self.graph {
            Some(graph) => graph,
            None => panic!(),
        }
    }

    pub(crate) fn graph_mut(&mut self) -> &mut Graph {
        match &mut self.graph {
            Some(graph) => graph,
            None => panic!(),
        }
    }

    pub(crate) fn take_graph(&mut self) -> Option<Graph> {
        self.graph.take()
    }

    pub(crate) fn out_ids(&self) -> &Vec<TensorId> {
        &self.out_ids
    }

    fn constant(&mut self, tensor: Tensor) -> TensorId {
        self.graph_mut().constant(tensor)
    }
}
