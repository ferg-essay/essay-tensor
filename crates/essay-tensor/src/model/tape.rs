use std::{cell::RefCell, collections::HashMap};

use crate::{Tensor, tensor::NodeId};

use super::{Var, NodeOp};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TensorId(pub usize);

pub struct Tape {
    tensors: Vec<Option<Tensor>>,
    tail: Option<Tensor>,

    var_map: HashMap<String, TensorId>,

    nodes: Vec<NodeOp>,
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
        self.nodes.len()
    }

    pub fn with(
        fun: impl FnOnce() -> Result<Tensor, TapeError>
    ) ->Result<Tape, TapeError> {
        let tape = Self {
            tensors: Default::default(),
            tail: None,

            var_map: Default::default(),
            nodes: Default::default(),
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
                Some(tape) => Some(tape.alloc_id_inner()),
                None => None,
            }
        })
    }

    fn alloc_id_inner(&mut self) -> TensorId {
        let id = TensorId(self.nodes.len());

        assert_eq!(self.tensors.len(), self.nodes.len(), 
            "alloc_id with mismatch graph and tensor");

        self.nodes.push(NodeOp::None);

        self.tensors.push(None);

        id
    }

    pub fn get_graph(&self, id: TensorId) -> &NodeOp {
        &self.nodes[id.index()]
    }

    pub(crate) fn set_graph(id: TensorId, node: NodeOp) {
        TAPE.with(|f| {
            if let Some(tape) = f.borrow_mut().as_mut() {
                tape.nodes[id.index()] = node;
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
                tape.set_tensor_inner(id, tensor);
            }
        })
    }

    fn set_tensor_inner(&mut self, id: TensorId, tensor: Tensor) {
        self.tensors[id.index()] = Some(tensor);
    }

    pub fn var(name: &str) -> TensorId {
        TAPE.with(|f| {
            if let Some(tape) = f.borrow_mut().as_mut() {
                tape.var_inner(name)
            } else {
                panic!("Tape::var without context")
            }
        })
    }

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

    pub fn set_var(name: &str, tensor: &Tensor) {
        TAPE.with(|f| {
            if let Some(tape) = f.borrow_mut().as_mut() {
                let id = tape.var_inner(name);

                if tape.tensors[id.index()].is_none() {
                    tape.tensors[id.index()] = Some(tensor.clone_id(id));
                }
            }
        })
    }

    pub fn gradient(
        &self, 
        var: &Var
    ) -> Tensor {
        if let Some(loss) = &self.tail {
            let id = match loss.op() {
                NodeId::Id(id) => id,
                _ => panic!("gradient needs saved graph in loss tensor"),
            };
        
            println!("var {:?}", var);
            println!("loss {:?}", loss);

            match &self.nodes[id.index()] {
                NodeOp::Op(op, args) => {
                    op.gradient(1, &self.to_args(args))
                }
                node => panic!("unsupported node for gradient {:?}", node)
            }
        } else {
            panic!("unset loss function")
        }
    }

    fn to_args(&self, args: &[TensorId]) -> Vec<&Tensor> {
        args.iter().map(|id| {
            match &self.tensors[id.index()] {
                Some(tensor) => tensor,
                None => panic!("unassigned tensor {:?}", id),
            }
        }).collect()
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
    use crate::model::{Var, TensorId};
    use crate::model::tape::Tape;
    use crate::{Tensor, random::uniform};
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
    fn test_mse() {
        let a = Var::new("z", tensor!(1.));

        let tape = Tape::with(|| {
            let y : Tensor = tensor!(0.0);
            //let loss: Tensor = (&a - &y) * (&a - &y);
            let loss: Tensor = a.l2_loss(&y);

            println!("z {:#?}", &a);
            println!("y {:#?}", &y);
            println!("loss {:#?}", &loss);

            Ok(loss)
        }).unwrap();

        println!("tensor len {:?}", tape.len());
        for i in 0..tape.len() {
            println!("  tensor[{:?}] {:?}", i, tape.get_tensor(TensorId(i)));
        }
        println!("graph len {:?}", tape.graph_len());
        for i in 0..tape.graph_len() {
            println!("  graph[{:?}] {:?}", i, tape.get_graph(TensorId(i)));
        }
        let dz = tape.gradient(&a);
        println!("dz {:#?}", &dz);
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