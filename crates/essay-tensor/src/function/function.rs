use super::{Var, Graph, TensorCache, Tape};
use crate::tensor::{TensorId, Tensors};

pub struct Function<In: Tensors, Out: Tensors> {
    _vars: Vec<(Var, TensorId)>,
    fun: Box<dyn Fn(&Graph, In, &TensorCache) -> Out>,

    graph: Graph,
}

impl<In, Out> Function<In, Out>
where
    In: Tensors<Item=In>,
    Out: Tensors<Item=Out>,
{
    pub fn compile<F>(input: In, fun: F) -> Function<In, Out>
    where
        F: FnOnce(In) -> Out,
    {
        let mut tape = Tape::build(input, fun);

        let out_ids = tape.out_ids().clone();

        Self {
            _vars: Default::default(),
            fun: Box::new(move |graph: &Graph, input, fwd_tensors| { 
                let mut out = graph.tensors().clone();

                In::set_arg(&mut out, 0, &input);

                graph.apply(&mut out, fwd_tensors);

                let mut index = 0;
                let value = Out::make_out(&out, &out_ids, &mut index);

                value
            }),

            graph: tape.take_graph().unwrap(),
        }
    }

    pub fn call(&self, input: In) -> Out {
        let tensors = self.graph.tensors().clone();

        (self.fun)(&self.graph, input, &tensors)
    }
}

#[cfg(test)]
mod test {
    use log::LevelFilter;

    use crate::function::{Tape};

    use crate::{
        function::{function::Function, Var},
        tensor::{TensorId},
        tensor,
        Tensor,
    };

    #[test]
    fn test_alloc() {
        assert_eq!(Tape::alloc_id(), None);

        let _module = Function::compile((), |()| {
            assert_eq!(Tape::alloc_id(), Some(TensorId(0)));
            assert_eq!(Tape::alloc_id(), Some(TensorId(1)));

            tensor!(0.)
        });
        
        assert_eq!(Tape::alloc_id(), None);
    }

    #[test]
    fn var() {
        let a = Var::new("a", tensor!([[1.]]));
        let x = Var::new("x", tensor!([2.]));

        let m_a = Function::compile((), 
            |_| a.tensor().clone()
        );

        let value = m_a.call(());
        assert_eq!(value, tensor!([[1.]]));

        let m_x = Function::compile((), 
            |_| x.tensor().clone()
        );

        let value = m_x.call(());
        assert_eq!(value, tensor!([2.]));
        // let train = module.train(());
    }

    #[test]
    fn binop_mul() {
        let a = Var::new("a", tensor!([1., 2., 3.]));

        let m_a = Function::compile((), 
        |_| &a * tensor!(2.)
        );

        let value = m_a.call(());
        assert_eq!(value, tensor!([2., 4., 6.]));
    }

    #[test]
    fn single_arg() {
        env_logger::builder().filter_level(LevelFilter::Debug).init();

        let a = Var::new("a", tensor!([1., 2., 3.]));

        let m_a = Function::compile(
            tensor!([2., 1., 2.]), 
            |x| &a * &x
        );

        let value = m_a.call(tensor!([2., 1., 2.]));
        assert_eq!(value, tensor!([2., 2., 6.]));

        let value = m_a.call(tensor!([1., 1., 1.]));
        assert_eq!(value, tensor!([1., 2., 3.]));
    }

    #[test]
    fn dual_arg() {
        // env_logger::builder().filter_level(LevelFilter::Debug).init();

        let m_a = Function::compile(
            (tensor!([1., 1.]), tensor!([1., 1.])),
            |(x, y)| &x - &y
        );

        let value = m_a.call((tensor!([2., 1.]), tensor!([1., 2.])));
        assert_eq!(value, tensor!([1., -1.]));

        let value = m_a.call((tensor!([1., 2.]), tensor!([2., 1.])));
        assert_eq!(value, tensor!([-1., 1.]));
    }

    #[test]
    fn dual_out() {
        // env_logger::builder().filter_level(LevelFilter::Debug).init();

        let m_a = Function::compile(
            (tensor!([1., 1.]), tensor!([1., 1.])),
            |(x, y)| (y.clone(), x.clone())
        );

        let (y, x) = m_a.call((tensor!([2., 1.]), tensor!([1., 2.])));
        assert_eq!(x, tensor!([2., 1.]));
        assert_eq!(y, tensor!([1., 2.]));
    }
}
