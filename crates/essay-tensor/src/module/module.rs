use std::{cell::RefCell};

use crate::Tensor;

use super::{Var, TensorId, NodeOp, backprop::backprop_graph, Graph, TensorCache};

pub struct Module<In: Bundle, Out: Bundle> {
    _vars: Vec<(Var, TensorId)>,
    fun: Box<dyn Fn(&Graph, In, &TensorCache) -> Out>,

    graph: Graph,
    _tensors: Vec<Option<Tensor>>,
    gradients: Vec<(String, Graph)>,
}

pub trait Bundle : Clone {}

pub struct Train<'a, In: Bundle, Out: Bundle> {
    module: &'a Module<In, Out>,
    tensors: TensorCache,
    out: Out,

}

pub struct ModuleTape {
    _args: Vec<TensorId>,
    _vars: Vec<(Var, TensorId)>,
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
            _args: Default::default(),
            _vars: Default::default(),
            tensors: Default::default(),
            tail: None,

            graph: Default::default(),
        };

        // TODO: add RALL guard?
        TAPE.with(|f| {
            assert!(f.borrow().is_none());
            f.borrow_mut().replace(tape);
        });

        fun(init);

        let tape = TAPE.with(|f| f.borrow_mut().take().unwrap());

        let graph = tape.graph;

        Self {
            _vars: Default::default(),
            fun: Box::new(|graph: &Graph, _input, tensors| { 
                let out = graph.apply(tensors, &[]);
                out.last()
            }),

            graph: graph,
            _tensors: tape.tensors,
            gradients: Default::default(),
        }
    }

    pub fn training(
        self, 
        vars: &[&Var],
    ) -> Self {
        let mut graphs : Vec<(String, Graph)> = Vec::new();

        for var in vars {
            let id = self.graph.get_var(var);

            let graph = backprop_graph(&self.graph, id);

            graphs.push((var.name().to_string(), graph));
        } 

        Self {
            gradients: graphs,
            ..self
        }
    }

    pub fn eval(&self, input: In) -> Tensor {
        let tensors = self.graph.tensors().clone();

        (self.fun)(&self.graph, input, &tensors)
    }

    pub fn train(&self, input: In) -> Train<In, Tensor> {
        let tensors = self.graph.tensors().clone();

        let out = (self.fun)(&self.graph, input, &tensors);

        Train {
            module: self,
            out,
            tensors,
        }
    }
}

impl<In:Bundle,Out:Bundle> Train<'_, In, Out> {
    pub fn value(&self) -> Out {
        self.out.clone()
    }

    pub fn gradient(&self, var :&Var) -> Tensor {
        for (grad_var, grad_graph) in &self.module.gradients {
            if var.name() == grad_var {
                let tensors = grad_graph.apply(&self.tensors, &[]);
        
                return tensors.last()
            }
        }

        panic!("{:?} is an unknown gradient", var);
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
            _args: Default::default(),
            _vars: Default::default(),

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

    pub fn var_inner(&mut self, _new_var: &str) -> TensorId {
        todo!()
        //self.graph.var(new_var)
        /*
        for (var, id) in &self.vars {
            if var.name() == new_var {
                return *id
            }
        }

        let tensor_id = self.graph.var(new_var, new_var.tensor_raw());

        self.vars.push((new_var.clone(), tensor_id));

        tensor_id
        */
    }

    pub fn find_var_inner(&mut self, new_var: &str) -> TensorId {
        self.graph.find_var(new_var)
    }

    pub fn set_var_inner(&mut self, name: &str, tensor: &Tensor) -> TensorId {
        self.graph.var(name, tensor)
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
    use crate::module::{TensorId, ModuleTape};

    use crate::{
        module::{module::Module, Var},
        tensor, Tensor,
    };

    #[test]
    fn test_alloc() {
        assert_eq!(ModuleTape::alloc_id(), None);

        let _module = Module::build((), |()| {
            assert_eq!(ModuleTape::alloc_id(), Some(TensorId(0)));
            assert_eq!(ModuleTape::alloc_id(), Some(TensorId(1)));

            tensor!(0.)
        });
        
        assert_eq!(ModuleTape::alloc_id(), None);
    }

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
        |_| x.into()
        );

        let value = m_x.eval(());
        assert_eq!(value, tensor!([2.]));
        // let train = module.train(());
    }

    #[test]
    fn binop_mul() {
        let a = Var::new("a", tensor!([1., 2., 3.]));

        let m_a = Module::build((), 
        |_| &a * tensor!(2.)
        );

        let value = m_a.eval(());
        assert_eq!(value, tensor!([2., 4., 6.]));
    }

    #[test]
    fn grad_sub() {
        //env_logger::builder().filter_level(LevelFilter::Debug).init();

        let a = Var::new("a", tensor!([1., 2., 3.]));

        let m_a = Module::build((), 
        |_| tensor!(2.) - &a
        ).training(&[&a]);

        let value = m_a.eval(());
        assert_eq!(value, tensor!([1., 0., -1.]));

        let train = m_a.train(());

        assert_eq!(train.value(), tensor!([1., 0., -1.]));
        assert_eq!(train.gradient(&a), tensor!([-1., -1., -1.]));
    }

    #[test]
    fn grad_mul() {
        //env_logger::builder().filter_level(LevelFilter::Debug).init();

        let a = Var::new("a", tensor!([1., 2., 3.]));

        let m_a = Module::build((), 
        |_| tensor!(2.) * &a
        ).training(&[&a]);

        let value = m_a.eval(());
        assert_eq!(value, tensor!([2., 4., 6.]));

        let train = m_a.train(());

        assert_eq!(train.value(), tensor!([2., 4., 6.]));
        assert_eq!(train.gradient(&a), tensor!([2., 2., 2.]));
    }

    #[test]
    fn test_var() {
        let x = Var::new("x", tensor!(0.));

        let module = Module::build((), |()| {
            x.tensor().clone()
        }).training(&[&x]);
        let train = module.train(());

        assert_eq!(train.gradient(&x), tensor!(1.));

        let x = Var::new("x", tensor!([[0., 2.], [10., 11.]]));
        let module = Module::build((), |()| {
            x.tensor().clone()
        }).training(&[&x]);
        let train = module.train(());

        assert_eq!(train.gradient(&x), tensor!([[1., 1.], [1., 1.]]));
    }

    #[test]
    fn test_l2_loss() {
        let a = Var::new("a", tensor!(2.));

        let module = Module::build((), |()| {
            a.l2_loss()
        }).training(&[&a]);
        let train = module.train(());
        let da = train.gradient(&a);
        assert_eq!(da, tensor!(2.0));

        let a = Var::new("a", tensor!(3.));
        let y = Var::new("y", tensor!(0.));

        let module = Module::build((), |()| {
            let loss: Tensor = (&a - &y).l2_loss();
            assert_eq!(loss, tensor!(4.5));
            loss
        }).training(&[&a, &y]);
        let train = module.train(());

        let da = train.gradient(&a);
        assert_eq!(da, tensor!(3.0));

        let dy = train.gradient(&y);
        assert_eq!(dy, tensor!(-3.0));

        let a = Var::new("a", tensor!([1., 2.]));

        let module = Module::build((), |()| {
            let loss: Tensor = a.l2_loss();

            assert_eq!(&loss, &tensor!(1.25));

            loss
        }).training(&[&a]);
        let train = module.train(());

        let da = train.gradient(&a);
        assert_eq!(da, tensor!([1.0, 2.0]));
    }

    #[test]
    fn test_sq_rank_0() {
        let a = Var::new("a", tensor!(1.));
        let x = Var::new("x", tensor!(0.));

        let module = Module::build((), |()| {
            let loss: Tensor = (&a - &x) * (&a - &x);

            loss
        }).training(&[&a, &x]);
        let train = module.train(());

        assert_eq!(train.gradient(&a), tensor!(2.));
        assert_eq!(train.gradient(&x), tensor!(-2.));
    }

    #[test]
    fn test_sq_rank_1() {
        let a = Var::new("a", tensor!([1., 2.]));
        let x = Var::new("x", tensor!([0., 0.]));

        let module = Module::build((), |()| {
            let loss: Tensor = (&a - &x) * (&a - &x);

            loss
        }).training(&[&a, &x]);
        let train = module.train(());

        assert_eq!(train.gradient(&a), tensor!(2.));
        assert_eq!(train.gradient(&x), tensor!(-2.));
    }

    #[test]
    fn test_sq_rank_2() {
        let a = Var::new("a", tensor!([[1., 2.], [3., 4.]]));
        let x = Var::new("x", tensor!([[0., 1.], [0., 2.]]));

        let module = Module::build((), |()| {
            let loss: Tensor = (&a - &x) * (&a - &x);

            loss
        }).training(&[&a, &x]);
        let train = module.train(());

        assert_eq!(train.gradient(&a), tensor!(2.));
        assert_eq!(train.gradient(&x), tensor!(-2.));

        let module = Module::build((), |()| {
            (&a - &x) * (&a - &x)
        }).training(&[&a, &x]);
        let train = module.train(());

        assert_eq!(train.gradient(&a), tensor!([[2., 2.], [6., 4.]]));
        assert_eq!(train.gradient(&x), tensor!([[-2., -2.], [-6., -4.]]));
    }

    #[test]
    fn test_matvec() {
        /*
        let w = Var::new("w", tensor!(0.5));
        let b = Var::new("b", tensor!(0.5));

        let mut tape = Tape::new();
        let w_t = tape.var(&w);
        let b_t = tape.var(&b);

        let x = tensor!(0.0);

        let z = x.clone() * w_t.clone() + b_t;

        let y : Tensor = tensor!(2.0) * x + 1.0.into();
        let loss: Tensor = z.mean_square_error(&y);

        println!("w_t {:#?}", &w_t);
        println!("{:#?} loss {:#?}", &z, &loss);

        let dw = tape.gradient(&loss, &w);

        println!("w_t {:#?}", &w_t);
        println!("{:#?} loss {:#?}", &z, &loss);
        println!("dw {:#?}", &dw);
        */
    }
}
