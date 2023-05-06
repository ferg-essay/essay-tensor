use std::{cell::RefCell, collections::{HashMap, HashSet}};

use crate::{Tensor, tensor::{NodeId}};

use super::{Var, NodeOp, graph::{ArgTrace, BackTrace, Graph}};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(pub usize);

pub struct Tape {
    tensors: Vec<Option<Tensor>>,
    tail: Option<Tensor>,

    graph: Graph,
    /*
    var_map: HashMap<String, TensorId>,

    nodes: Vec<NodeOp>,
    */
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

    /*
    fn alloc_id_inner(&mut self) -> TensorId {
        let id = TensorId(self.nodes.len());

        assert_eq!(self.tensors.len(), self.nodes.len(), 
            "alloc_id with mismatch graph and tensor");

        self.nodes.push(NodeOp::None);

        self.tensors.push(None);

        id
    }
    */

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

    /*
    fn set_tensor_inner(&mut self, id: TensorId, tensor: Tensor) {
        while self.tensors.len () <= id.index() {
            self.tensors.push(None);
        }

        self.tensors[id.index()] = Some(tensor);
    }
    */

    pub fn var(name: &str) -> TensorId {
        TAPE.with(|f| {
            if let Some(tape) = f.borrow_mut().as_mut() {
                tape.graph.var(name)
            } else {
                panic!("Tape::var without context")
            }
        })
    }

    /*
    fn var_inner(&mut self, name: &str) -> TensorId {
        let len = self.len();
        let id = *self.var_map
            .entry(name.to_string())
            .or_insert(TensorId(len));

        if id.index() == len {
            self.tensors.push(None);
            self.nodes.push(NodeOp::Var(id, name.to_string()));
        }

        id
    }
    */

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

    /*
    fn build_backtrace(&self, target: TensorId) -> Option<BackTrace> {
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
                NodeOp::Op(_, args) => {
                    visited.insert(id);

                    let mut next_args = Vec::<NextTrace>::new();

                    for (i, arg) in args.iter().enumerate() {
                        if let Some(trace) =  self.build_backtrace_rec(target, *arg, visited) {
                            next_args.push(NextTrace(i, trace));
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

                }
            }
        }
    }
    */

    pub fn x_gradient(
        &self, 
        var: &Var
    ) -> Tensor {
        todo!();
        /*
        let id = self.var_map.get(var.name()).unwrap();
        let mut trace = self.build_backtrace(*id).unwrap();
        let mut partial = Tensor::from(1.);

        let mut trace_ptr = &trace;
        while let NodeOp::Op(op, args) = &self.nodes[trace_ptr.id.index()] {
            let mut next_partial = op.gradient(trace_ptr.args[0].0, &self.to_args(args));
            println!("  partial.0 {:?}", next_partial);

            for arg in &trace_ptr.args[1..] {
                let sub_partial = op.gradient(arg.0, &self.to_args(args));
                println!("    sub-partial.0 {:?}", sub_partial);
                next_partial = next_partial + sub_partial;
                println!("    => next-partial.0 {:?}", next_partial);
            }
            //}

            //partial = partial.matmul(next_partial);
            partial = partial * next_partial;

            trace_ptr = &trace_ptr.args[0].1;
        };

        partial
        */
    }

    pub fn gradient(
        &mut self, 
        var: &Var
    ) -> Tensor {
        let id = self.graph.get_var(var);
        //let trace = self.graph.build_backtrace(id).unwrap();

        let graph = self.graph.backtrace_graph(id);

        let tensors_in = self.graph.tensors();

        let tensors = graph.apply(&tensors_in, &[]);

        //self.gradient_rec(&trace, None)

        // None => Tensor::ones(self.get_tensor(*id).unwrap().shape())
        tensors.last()
    }

    fn gradient_rec(
        &self, 
        trace: &BackTrace,
        prev: Option<Tensor>,
    ) -> Tensor {
        todo!();
        /*
        match &self.graph.node(trace.id) { // nodes[trace.id.index()] {
            NodeOp::None => todo!(),
            NodeOp::Const(_) => todo!(),
            NodeOp::Var(id, name) => {
                match prev {
                    Some(prev) => prev,
                    None => Tensor::ones(self.get_tensor(*id).unwrap().shape())
                }
            },
            NodeOp::Op(id, op, args) => {
                let t_args = self.to_args(&args);

                let mut gradient: Option<Tensor> = None;

                for i in 0..trace.args.len() {
                    let partial = match &prev {
                        Some(prev) => op.gradient(trace.args[i].index(), &t_args, prev),
                        None => op.gradient_top(trace.args[i].index(), &t_args),

                    let next = self.gradient_rec(&trace.args[i].backtrace(), Some(partial));

                    gradient = Some(next + gradient)
                }

                gradient.unwrap()
            }
        }
        */
    }

    fn to_args(&self, args: &[TensorId]) -> Vec<&Tensor> {
        args.iter().map(|id| {
            match &self.tensors[id.index()] {
                Some(tensor) => tensor,
                None => panic!("unassigned tensor {:?}", id),
            }
        }).collect()
    }
    /*
    fn partial(&self, op: &dyn Op, index: usize, args: &[TensorId]) -> Tensor {
        let args = self.to_args(args);

        op.gradient(index, &args)
    }
    */
}

impl TensorId {
    #[inline]
    pub fn index(&self) -> usize {
        self.0
    }
}

#[cfg(test)]
mod test {
    use crate::model::{Var, TensorId};
    use crate::model::tape::Tape;
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
    fn test_mse() {
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
        let w = Var::new("w", tensor!(0.5));
        let b = Var::new("b", tensor!(0.5));
        /*
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