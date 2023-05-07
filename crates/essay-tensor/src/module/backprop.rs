use std::{collections::HashSet, any::type_name};

use crate::Tensor;

use super::{Graph, TensorId, NodeOp, IntoForward, BoxForwardOp, ForwardOp, EvalOp, TensorCache, graph::{IntoBack, BackOp, BoxBackOp}};


pub struct ArgTrace {
    arg_index: usize,
    backtrace: BackTrace
}

pub struct BackTrace {
    pub id: TensorId,
    pub args: Vec<ArgTrace>,
}

pub(crate) fn backprop_graph(forward: &Graph, target: TensorId) -> Graph {
    let backtrace = build_backtrace(forward, target).unwrap();

    let mut graph = Graph::default();

    let tail = forward.tensor(forward.tail_id()).unwrap();

    let tail = graph.constant(Tensor::ones(tail.shape()));

    backprop_graph_rec(forward, &mut graph, &backtrace, tail);

    graph
}

pub(crate) fn backprop_graph_rec(
    forward: &Graph,
    back: &mut Graph,
    backtrace: &BackTrace,
    prev: TensorId,
) {
    let len = backtrace.args.len();

    if len == 0 {
        node_backprop(forward, back, 0, backtrace.id, prev);
    }

    for arg in &backtrace.args {
        let back_arg = node_backprop(forward, back, arg.index(), backtrace.id, prev);

        backprop_graph_rec(forward, back, arg.backtrace(), back_arg);
    }
}

fn node_backprop(
    forward: &Graph,
    back: &mut Graph,
    i: usize,
    id: TensorId,
    prev: TensorId
) -> TensorId {
    match forward.node(id) {
        NodeOp::None => todo!(),
        NodeOp::Const(_id) => todo!(),
        NodeOp::Var(_, _) => {
            prev
        }
        NodeOp::Op(_, op, args) => {
            op.backprop(forward, back, i, args, prev)
        },
        NodeOp::BackConst(_, _) => panic!("BackConst is invalid when generating backtrace"),
        NodeOp::BackOp(_, _, _, _) => panic!("BackOp is invalid when generating backtrace"),
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
            NodeOp::BackOp(_, _, _, _) => {
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
        _prev: TensorId,
    ) -> TensorId {
        panic!("{} does not implement backprop", type_name::<Op>())
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

impl<Op> IntoBack for Op
where
    Op: Clone + BackOp
{
    fn to_op(&self) -> BoxBackOp {
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
    use crate::module::{Var};
    use crate::module::tape::Tape;
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
    fn test_l2_loss() {
        let x = Var::new("x", tensor!(2.));

        let mut tape = Tape::with(|| {
            let loss: Tensor = x.l2_loss();

            Ok(loss)
        }).unwrap();

        let dx = tape.gradient(&x);
        assert_eq!(dx, tensor!(2.0));

        let x = Var::new("x", tensor![3.]);

        let mut tape = Tape::with(|| {
            let loss: Tensor = x.l2_loss();

            Ok(loss)
        }).unwrap();

        let dx = tape.gradient(&x);
        assert_eq!(dx, tensor![3.]);

        let x = Var::new("x", tensor!([1., 2., 3.]));

        let mut tape = Tape::with(|| {
            let loss: Tensor = x.l2_loss();

            Ok(loss)
        }).unwrap();

        let dx = tape.gradient(&x);
        assert_eq!(dx, tensor!([1., 2., 3.]));
    }

    #[test]
    fn test_sub() {
        let x = Var::new("x", tensor!(3.));
        let y = Var::new("y", tensor!(1.));

        let mut tape = Tape::with(|| {
            let out: Tensor = &x - &y;
            assert_eq!(out, tensor!(2.));

            Ok(out)
        }).unwrap();

        let dx = tape.gradient(&x);
        assert_eq!(dx, tensor!(1.0));

        let dy = tape.gradient(&y);
        assert_eq!(dy, tensor!(-1.0));

        let x = Var::new("x", tensor!([4., 5., 7.]));
        let y = Var::new("y", tensor!([1., 2., 3.]));

        let mut tape = Tape::with(|| {
            let out: Tensor = &x - &y;
            assert_eq!(out, tensor!([3., 3., 4.]));

            Ok(out)
        }).unwrap();

        let dx = tape.gradient(&x);
        assert_eq!(dx, tensor!([1., 1., 1.]));

        let dy = tape.gradient(&y);
        assert_eq!(dy, tensor!([-1., -1., -1.]));
    }

    #[test]
    fn test_l2_loss_diff() {
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
        let x = Var::new("x", tensor!([0., 0.]));

        let mut tape = Tape::with(|| {
            let loss: Tensor = (&a - &x).l2_loss();

            assert_eq!(&loss, &tensor!(1.25));

            Ok(loss)
        }).unwrap();

        let da = tape.gradient(&a);
        assert_eq!(da, tensor!([1.0, 2.0]));

        let dx = tape.gradient(&y);
        assert_eq!(dx, tensor!([-1.0, -2.0]));
    }
}