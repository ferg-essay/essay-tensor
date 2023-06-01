use std::cell::RefCell;
use crate::{Tensor, tensor::{TensorId}};


use super::{Var, TensorCache, Expr, NodeOp, Tensors, model::ModelId};

pub struct Tape {
    _args: Vec<TensorId>,
    _vars: Vec<(Var, TensorId)>,
    tensors: TensorCache,
    //_tail: Option<Tensor>,
    out_ids: Vec<TensorId>,

    expr: Option<Expr>,
}

#[derive(Debug)]
pub enum TapeError {}

thread_local! {
    pub static TAPE: RefCell<Option<Tape>> = RefCell::new(None);
}

impl Tape {
    pub fn build<F, In, Out>(id: ModelId, init: In, fun: F) -> Tape
    where
        In: Tensors<Item=In>,
        Out: Tensors<Item=Out>,
        F: FnOnce(In::Item) -> Out,
    {
        let mut tape = Tape {
            _args: Default::default(),
            _vars: Default::default(),
            tensors: TensorCache::new(id),
            out_ids: Default::default(),

            expr: Some(Expr::new(id)),
        };

        let mut index = 0;
        // TODO: check that the clone and &Tensor work together
        let tensors_clone = tape.tensors().clone();
        let args = In::make_arg(&tensors_clone, &mut index);

        let arg_len = In::push_arg(tape.tensors_mut(), 0, &init);

        for arg_id in 0..arg_len {
            let arg_id = TensorId::new(id.index() as u32, arg_id as u32);

            let tensor = tape.tensors.get(arg_id).unwrap().clone();
            tape.arg(tensor);
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
        match &mut self.expr {
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

    pub(crate) fn set_tensor(tensor: Tensor) -> Tensor {
        if tensor.id().is_some() {
            TAPE.with(|f| {
                if let Some(tape) = f.borrow_mut().as_mut() {
                    tape.graph_mut().set_tensor(tensor.id(), tensor.clone());
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

    pub fn var(var: &Var) -> Tensor {
        TAPE.with(|f| {
            if let Some(tape) = f.borrow_mut().as_mut() {
                tape.var_inner(var)
            } else {
                var.tensor_raw()
            }
        })
    }

    pub fn var_inner(&mut self, var: &Var) -> Tensor {
        self.graph_mut().var(var)
    }

    pub(crate) fn tracked_vars(&self) -> &Vec<Var> {
        self.graph().tracked_vars()
    }

    pub(crate) fn tensors(&self) -> &TensorCache {
        &self.tensors
    }

    pub(crate) fn tensors_mut(&mut self) -> &mut TensorCache {
        &mut self.tensors
    }

    pub(crate) fn graph(&self) -> &Expr {
        match &self.expr {
            Some(graph) => graph,
            None => panic!(),
        }
    }

    pub(crate) fn graph_mut(&mut self) -> &mut Expr {
        match &mut self.expr {
            Some(graph) => graph,
            None => panic!(),
        }
    }

    pub(crate) fn take_graph(&mut self) -> Option<Expr> {
        self.expr.take()
    }

    pub(crate) fn out_ids(&self) -> &Vec<TensorId> {
        &self.out_ids
    }

    fn arg(&mut self, tensor: Tensor) -> TensorId {
        self.graph_mut().arg(tensor)
    }
}
