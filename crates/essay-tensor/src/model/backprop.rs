use std::{collections::HashSet, any::type_name};

use crate::Tensor;

use super::{Graph, TensorId, NodeOp, IntoForward, BoxForwardOp, ForwardOp, EvalOp, TensorCache};


pub struct ArgTrace {
    arg_index: usize,
    backtrace: BackTrace
}

pub struct BackTrace {
    pub id: TensorId,
    pub args: Vec<ArgTrace>,
}

pub(crate) fn backtrace_graph(forward: &Graph, target: TensorId) -> Graph {
    let backtrace = build_backtrace(forward, target).unwrap();

    let mut graph = Graph::default();

    backtrace_graph_rec(forward, &mut graph, &backtrace, None);

    assert!(graph.len() > 0, "backtrace produced empty graph");

    graph
}

pub(crate) fn backtrace_graph_rec(
    forward: &Graph,
    back: &mut Graph,
    backtrace: &BackTrace,
    prev: Option<TensorId>,
) {
    let len = backtrace.args.len();

    if len == 0 {
        node_backtrace(forward, back, 0, backtrace.id, prev);
    }

    for arg in &backtrace.args {
        let back_arg = node_backtrace(forward, back, arg.index(), backtrace.id, prev);

        backtrace_graph_rec(forward, back, arg.backtrace(), Some(back_arg));
    }
}

fn node_backtrace(
    forward: &Graph,
    graph: &mut Graph,
    i: usize,
    id: TensorId,
    prev: Option<TensorId>
) -> TensorId {
    match forward.node(id) {
        NodeOp::None => todo!(),
        NodeOp::Const(_id) => todo!(),
        NodeOp::Var(id, _) => {
            match prev {
                Some(prev) => {
                    prev
                },
                None => {
                    let id = graph.constant(Tensor::ones(forward.tensor(*id).unwrap().shape()));

                    id
                }
            }
        }
        NodeOp::Op(id, op, args) => {
            match prev {
                Some(prev) => {
                    op.backprop(forward, graph, i, args, *id, prev)
                },
                None => {
                    op.backprop_top(forward, graph, i, args, *id)
                }
            }
        },
        NodeOp::BackConst(_, _) => panic!("BackConst is invalid when generating backtrace"),
        NodeOp::BackOp(_, _, _) => panic!("BackConst is invalid when generating backtrace"),
    }
}

pub(crate) fn build_backtrace(graph: &Graph, target: TensorId) -> Option<BackTrace> {
    let tail_id = graph.tail_id();

    build_backtrace_rec(graph, target, tail_id, &mut HashSet::new())
}

fn build_backtrace_rec(
    graph: &Graph, 
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
        match &graph.node(id) {
            NodeOp::None => None,
            NodeOp::Const(_) => None,
            NodeOp::Var(_, _) => None,
            NodeOp::Op(_, _, args) => {
                visited.insert(id);

                let mut next_args = Vec::<ArgTrace>::new();

                for (i, arg) in args.iter().enumerate() {
                    if let Some(trace) =  build_backtrace_rec(graph, target, *arg, visited) {
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

#[cfg(test)]
mod test {
    use crate::model::{Var, TensorId};
    use crate::model::tape::Tape;
    use crate::{Tensor};
    use crate::prelude::{*};

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
}