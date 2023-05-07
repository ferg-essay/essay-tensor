use std::{cell::RefCell};

use crate::{Tensor};

use super::{Var, NodeOp, graph::{Graph}, backprop::backprop_graph};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(pub usize);

pub struct Tape {
    tensors: Vec<Option<Tensor>>,
    tail: Option<Tensor>,

    graph: Graph,
}

thread_local! {
    pub static TAPE: RefCell<Option<Tape>> = RefCell::new(None);
}

#[derive(Debug)]
pub enum TapeError {
}

impl Tape {
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn graph_len(&self) -> usize {
        self.graph.len()
    }

    pub fn with(
        fun: impl FnOnce() -> Result<Tensor, TapeError>
    ) ->Result<Tape, TapeError> {
        let tape = Self {
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

        let mut tape = TAPE.with(|f| {
            f.borrow_mut().take().unwrap()
        });

        tape.tail = Some(tail?);

        Ok(tape)
    }

    pub fn is_active() -> bool {
        TAPE.with(|f| {
            match f.borrow().as_ref() {
                Some(_) => true,
                None => false,
            }
        })
    }

    pub fn alloc_id() -> Option<TensorId> {
        TAPE.with(|f| {
            match f.borrow_mut().as_mut() {
                Some(tape) => Some(tape.graph.alloc_id()),
                None => None,
            }
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

    pub fn var(name: &str) -> TensorId {
        TAPE.with(|f| {
            if let Some(tape) = f.borrow_mut().as_mut() {
                tape.graph.var(name)
            } else {
                panic!("Tape::var without context")
            }
        })
    }

    pub fn set_var(name: &str, tensor: &Tensor) {
        TAPE.with(|f| {
            if let Some(tape) = f.borrow_mut().as_mut() {
                let id = tape.graph.var(name);

                tape.graph.set_tensor(id, tensor.clone());

                while tape.tensors.len() <= id.index() {
                    tape.tensors.push(None);
                }

                if tape.tensors[id.index()].is_none() {
                    tape.tensors[id.index()] = Some(tensor.clone_id(id));
                }
            }
        })
    }

    pub fn gradient(
        &mut self, 
        var: &Var
    ) -> Tensor {
        let id = self.graph.get_var(var);
        //let trace = self.graph.build_backtrace(id).unwrap();

        let graph = backprop_graph(&self.graph, id);

        let fwd_tensors = self.graph.tensors();

        let tensors = graph.apply(&fwd_tensors, &[]);

        tensors.last()
    }
}

impl TensorId {
    #[inline]
    pub fn index(&self) -> usize {
        self.0
    }
}

#[cfg(test)]
mod test {
    use crate::module::{Var, TensorId};
    use crate::module::tape::Tape;
    use crate::{Tensor};
    use crate::prelude::{*};

    #[test]
    fn test_alloc() {
        assert_eq!(Tape::alloc_id(), None);

        let _tape = Tape::with(|| {
            assert_eq!(Tape::alloc_id(), Some(TensorId(0)));
            assert_eq!(Tape::alloc_id(), Some(TensorId(1)));

            let loss = tensor!(0.);

            Ok(loss)
        }).unwrap();
        
        assert_eq!(Tape::alloc_id(), None);
    }

    #[test]
    fn test_var() {
        let x = Var::new("x", tensor!(0.));

        let mut tape = Tape::with(|| {
            Ok(x.tensor().clone())
        }).unwrap();

        assert_eq!(tape.gradient(&x), tensor!(1.));

        let x = Var::new("x", tensor!([[0., 2.], [10., 11.]]));

        let mut tape = Tape::with(|| {
            Ok(x.tensor().clone())
        }).unwrap();

        assert_eq!(tape.gradient(&x), tensor!([[1., 1.], [1., 1.]]));
    }

    #[test]
    fn test_l2_loss() {
        let a = Var::new("a", tensor!(2.));

        let mut tape = Tape::with(|| {
            let loss: Tensor = a.l2_loss();

            Ok(loss)
        }).unwrap();

        let dz = tape.gradient(&a);
        assert_eq!(dz, tensor!(2.0));

        let a = Var::new("a", tensor!(3.));
        let y = Var::new("y", tensor!(0.));

        let mut tape = Tape::with(|| {
            let loss: Tensor = (&a - &y).l2_loss();
            assert_eq!(loss, tensor!(4.5));
            Ok(loss)
        }).unwrap();

        let da = tape.gradient(&a);
        assert_eq!(da, tensor!(3.0));

        let dy = tape.gradient(&y);
        assert_eq!(dy, tensor!(-3.0));

        let a = Var::new("a", tensor!([1., 2.]));

        let mut tape = Tape::with(|| {
            let loss: Tensor = a.l2_loss();

            assert_eq!(&loss, &tensor!(1.25));

            Ok(loss)
        }).unwrap();

        let dz = tape.gradient(&a);
        assert_eq!(dz, tensor!([1.0, 2.0]));
    }

    #[test]
    fn test_sq_rank_0() {
        let a = Var::new("a", tensor!(1.));
        let x = Var::new("x", tensor!(0.));

        let mut tape = Tape::with(|| {
            let loss: Tensor = (&a - &x) * (&a - &x);

            Ok(loss)
        }).unwrap();

        assert_eq!(tape.gradient(&a), tensor!(2.));
        assert_eq!(tape.gradient(&x), tensor!(-2.));
    }

    #[test]
    fn test_sq_rank_1() {
        let a = Var::new("a", tensor!([1., 2.]));
        let x = Var::new("x", tensor!([0., 0.]));

        let mut tape = Tape::with(|| {
            let loss: Tensor = (&a - &x) * (&a - &x);

            Ok(loss)
        }).unwrap();

        assert_eq!(tape.gradient(&a), tensor!([2., 4.]));
        assert_eq!(tape.gradient(&x), tensor!([-2., -4.]));
    }

    #[test]
    fn test_sq_rank_2() {
        let a = Var::new("a", tensor!([[1., 2.], [3., 4.]]));
        let x = Var::new("x", tensor!([[0., 1.], [0., 2.]]));

        let mut tape = Tape::with(|| {
            let loss: Tensor = (&a - &x) * (&a - &x);

            Ok(loss)
        }).unwrap();

        assert_eq!(tape.gradient(&a), tensor!([[2., 2.], [6., 4.]]));
        assert_eq!(tape.gradient(&x), tensor!([[-2., -2.], [-6., -4.]]));
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