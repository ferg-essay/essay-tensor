use super::{Var, Expr, TensorCache, Tape, Tensors, model::ModelId};
use crate::tensor::{TensorId};

pub struct Function<In: Tensors, Out: Tensors> {
    fun: Box<dyn Fn(&Expr, In, &TensorCache) -> Out>,

    expr: Expr,
}

impl<In, Out> Function<In, Out>
where
    In: Tensors<Item=In>,
    Out: Tensors<Item=Out>,
{
    pub fn compile<F>(input: In, fun: F) -> Function<In, Out>
    where
        F: FnOnce(In::Item) -> Out,
    {
        let id = ModelId::alloc();

        let mut tape = Tape::build(id, input, fun);

        let out_ids = tape.out_ids().clone();

        Self {
            fun: Box::new(move |expr: &Expr, input, fwd_tensors| { 
                let mut out = expr.tensors().clone();

                In::set_arg(&mut out, 0, &input);

                expr.call(&mut out, fwd_tensors);

                let mut index = 0;
                let value = Out::make_out(&out, &out_ids, &mut index);

                value
            }),

            expr: tape.take_graph().unwrap(),
        }
    }

    pub fn call(&self, input: In) -> Out {
        let tensors = self.expr.tensors().clone();

        (self.fun)(&self.expr, input, &tensors)
    }
}

#[cfg(test)]
mod test {
    use log::LevelFilter;

    use crate::model::{Tape};

    use crate::{
        model::{function::Function, Var},
        tensor::{TensorId},
        tensor,
        Tensor,
    };

    #[test]
    fn test_alloc() {
        assert_eq!(Tape::alloc_id(), None);

        let _module = Function::compile((), |()| {
            assert_eq!(Tape::alloc_id(), Some(TensorId::new(0, 0)));
            assert_eq!(Tape::alloc_id(), Some(TensorId::new(0, 1)));

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
            |x| &a * x
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
            |(x, y)| x - y
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
