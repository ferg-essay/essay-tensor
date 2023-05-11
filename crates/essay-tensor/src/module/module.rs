use crate::{Tensor, tensor::NodeId};

use super::{Var, TensorId, backprop::backprop_graph, Graph, TensorCache, Tape, Bundle};

pub struct Module<In: Bundle, Out: Bundle> {
    _vars: Vec<(Var, TensorId)>,
    fun: Box<dyn Fn(&Graph, In, &TensorCache) -> Out>,

    graph: Graph,
    _tensors: TensorCache,
    gradients: Vec<(String, Graph)>,
}

pub struct Train<'a, In: Bundle<Item=In>, Out: Bundle<Item=Out>> {
    module: &'a Module<In, Out>,
    tensors: TensorCache,
    out: Out,

}

impl<In: Bundle<Item=In>, Out: Bundle<Item=Out>> Module<In, Out> {
    pub fn build<F>(init: In, fun: F) -> Module<In, Out>
    where
        F: FnOnce(In) -> Out,
    {
        /*
        let mut tape = Tape {
            _args: Default::default(),
            _vars: Default::default(),
            tensors: Default::default(),
            _tail: None,

            graph: Default::default(),
        };

        let len = In::push_arg(&mut tape.tensors(), 0, &init);

        // Tensors in args now have their id set.
        let mut index = 0;
        let args = In::make_arg(&tape.tensors(), &mut index);

        for id in 0..len {
            tape.graph().constant(tape.get_tensor(TensorId(id)).unwrap().clone());
        }

        // TODO: add RALL guard?
        TAPE.with(|f| {
            assert!(f.borrow().is_none());
            f.borrow_mut().replace(tape);
        });


        let out = fun(args);
        let mut out_ids : Vec<TensorId> = Vec::new();
        Out::out_ids(&mut out_ids, &out);

        let tape = TAPE.with(|f| f.borrow_mut().take().unwrap());
        */

        let mut tape = Tape::build(init, fun);

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
            _tensors: tape.tensors().clone(),
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

    pub fn eval(&self, input: In) -> Out {
        let tensors = self.graph.tensors().clone();

        (self.fun)(&self.graph, input, &tensors)
    }

    pub fn train(&self, input: In) -> Train<In, Out> {
        let tensors = self.graph.tensors().clone();

        let out = (self.fun)(&self.graph, input, &tensors);

        Train {
            module: self,
            out,
            tensors,
        }
    }
}

impl<In:Bundle<Item=In>,Out:Bundle<Item=Out>> Train<'_, In, Out> {
    pub fn value(&self) -> Out {
        self.out.clone()
    }

    pub fn gradient(&self, var :&Var) -> Tensor {
        for (grad_var, grad_graph) in &self.module.gradients {
            if var.name() == grad_var {
                let mut out = grad_graph.tensors().clone();

                grad_graph.apply(&mut out, &self.tensors);
        
                return out.last()
            }
        }

        panic!("{:?} is an unknown gradient", var);
    }

}

#[cfg(test)]
mod test {
    use log::LevelFilter;

    use crate::module::{TensorId, Tape};

    use crate::{
        module::{module::Module, Var},
        tensor, Tensor,
    };

    #[test]
    fn test_alloc() {
        assert_eq!(Tape::alloc_id(), None);

        let _module = Module::build((), |()| {
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
    fn single_arg() {
        env_logger::builder().filter_level(LevelFilter::Debug).init();

        let a = Var::new("a", tensor!([1., 2., 3.]));

        let m_a = Module::build(
            tensor!([2., 1., 2.]), 
            |x| &a * &x
        );

        let value = m_a.eval(tensor!([2., 1., 2.]));
        assert_eq!(value, tensor!([2., 2., 6.]));

        let value = m_a.eval(tensor!([1., 1., 1.]));
        assert_eq!(value, tensor!([1., 2., 3.]));
    }

    #[test]
    fn dual_arg() {
        // env_logger::builder().filter_level(LevelFilter::Debug).init();

        let m_a = Module::build(
            (tensor!([1., 1.]), tensor!([1., 1.])),
            |(x, y)| &x - &y
        );

        let value = m_a.eval((tensor!([2., 1.]), tensor!([1., 2.])));
        assert_eq!(value, tensor!([1., -1.]));

        let value = m_a.eval((tensor!([1., 2.]), tensor!([2., 1.])));
        assert_eq!(value, tensor!([-1., 1.]));
    }

    #[test]
    fn dual_out() {
        // env_logger::builder().filter_level(LevelFilter::Debug).init();

        let m_a = Module::build(
            (tensor!([1., 1.]), tensor!([1., 1.])),
            |(x, y)| (y.clone(), x.clone())
        );

        let (y, x) = m_a.eval((tensor!([2., 1.]), tensor!([1., 2.])));
        assert_eq!(x, tensor!([2., 1.]));
        assert_eq!(y, tensor!([1., 2.]));
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
