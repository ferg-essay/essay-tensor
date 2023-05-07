use std::{collections::HashMap, cell::RefCell};

use crate::Tensor;

use super::{Var, TensorId, NodeOp, backprop::backprop_graph, Graph};

pub trait Bundle {}

pub struct Train<'a, In: Bundle, Out: Bundle> {
    out: Out,
    module: &'a Module<In, Out>,
}

pub struct Module<In: Bundle, Out: Bundle> {
    vars: Vec<(Var, TensorId)>,
    fun: Box<dyn Fn(&Graph, In) -> Out>,

    graph: Graph,
    tensors: Vec<Option<Tensor>>,
}

pub struct ModuleTape {
    args: Vec<TensorId>,
    vars: Vec<(Var, TensorId)>,
    tensors: Vec<Option<Tensor>>,
    tail: Option<Tensor>,

    graph: Graph,
}

thread_local! {
    pub static TAPE: RefCell<Option<ModuleTape>>  = RefCell::new(None);
}

#[derive(Debug)]
pub enum TapeError {}

impl<In: Bundle> Module<In, Tensor> {
    pub fn build<F>(init: In, fun: F) -> Module<In, Tensor>
    where
        F: FnOnce(In) -> Tensor,
    {
        let tape = ModuleTape {
            args: Default::default(),
            vars: Default::default(),
            tensors: Default::default(),
            tail: None,

            graph: Default::default(),
        };

        // TODO: add RALL guard?
        TAPE.with(|f| {
            assert!(f.borrow().is_none());
            f.borrow_mut().replace(tape);
        });

        let tail = fun(init);

        let mut tape = TAPE.with(|f| f.borrow_mut().take().unwrap());

        let graph = tape.graph;

        Self {
            vars: Default::default(),
            fun: Box::new(|graph: &Graph, In| { 
                let tensors = graph.tensors().clone();
                let out = graph.apply(graph.tensors(), &[]);
                out.last()
            }),

            graph: graph,
            tensors: tape.tensors,
        }
    }

    pub fn eval(&self, input: In) -> Tensor {
        (self.fun)(&self.graph, input)
    }

    pub fn train(&self, input: In) -> Train<In, Tensor> {
        todo!()
    }
}

impl ModuleTape {
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn graph_len(&self) -> usize {
        self.graph.len()
    }

    pub fn with(fun: impl FnOnce() -> Result<Tensor, TapeError>) -> Result<ModuleTape, TapeError> {
        let tape = Self {
            args: Default::default(),
            vars: Default::default(),

            tensors: Default::default(),
            tail: None,

            graph: Default::default(),
        };

        // TODO: add RALL guard?
        TAPE.with(|f| {
            assert!(f.borrow().is_none());
            f.borrow_mut().replace(tape);
        });

        let tail = fun();

        let mut tape = TAPE.with(|f| f.borrow_mut().take().unwrap());

        tape.tail = Some(tail?);

        Ok(tape)
    }

    pub fn is_active() -> bool {
        TAPE.with(|f| match f.borrow().as_ref() {
            Some(_) => true,
            None => false,
        })
    }

    pub fn alloc_id() -> Option<TensorId> {
        TAPE.with(|f| match f.borrow_mut().as_mut() {
            Some(tape) => Some(tape.graph.alloc_id()),
            None => None,
        })
    }

    pub fn node(&self, id: TensorId) -> &NodeOp {
        &self.graph.node(id) // nodes[id.index()]
    }

    pub(crate) fn set_node(id: TensorId, node: NodeOp) {
        TAPE.with(|f| {
            if let Some(tape) = f.borrow_mut().as_mut() {
                tape.graph.set_node(id, node);
                // tape.nodes[id.index()] = node;
            } else {
                panic!("call set_graph with no active tape");
            }
        })
    }

    pub fn get_tensor(&self, id: TensorId) -> Option<&Tensor> {
        match &self.tensors[id.index()] {
            Some(tensor) => Some(tensor),
            None => None,
        }
    }

    pub(crate) fn set_tensor(id: TensorId, tensor: Tensor) {
        TAPE.with(|f| {
            if let Some(tape) = f.borrow_mut().as_mut() {
                tape.graph.set_tensor(id, tensor);
            }
        })
    }

    pub fn var(var: &Var) -> TensorId {
        TAPE.with(|f| {
            if let Some(tape) = f.borrow_mut().as_mut() {
                tape.var_inner(var)
            } else {
                panic!("Tape::var without context")
            }
        })
    }

    pub fn var_inner(&mut self, new_var: &Var) -> TensorId {
        for (var, id) in &self.vars {
            if var.name() == new_var.name() {
                return *id
            }
        }

        let tensor_id = self.graph.var(new_var.name(), new_var.tensor_raw());

        self.vars.push((new_var.clone(), tensor_id));

        tensor_id
    }

    pub fn gradient(&mut self, var: &Var) -> Tensor {
        let id = self.graph.get_var(var);
        //let trace = self.graph.build_backtrace(id).unwrap();

        let graph = backprop_graph(&self.graph, id);

        let fwd_tensors = self.graph.tensors();

        let tensors = graph.apply(&fwd_tensors, &[]);

        tensors.last()
    }
}

impl Bundle for Tensor {}

impl Bundle for () {}

impl Bundle for (Tensor, Tensor) {}

#[cfg(test)]
mod test {
    use crate::{
        module::{module::Module, Tape, Var},
        tensor, Tensor,
    };

    #[test]
    fn var() {
        let a = Var::new("a", tensor!([[1.]]));
        let x = Var::new("x", tensor!([2.]));

        let m_a = Module::build((), 
        |_| a.tensor().clone()
        );

        let value = m_a.eval(());
        assert_eq!(value, tensor!([[1.]]));

        let m_x = Module::build((), 
        |_| x.tensor().clone()
        );

        let value = m_x.eval(());
        assert_eq!(value, tensor!([2.]));
        // let train = module.train(());
    }
}
