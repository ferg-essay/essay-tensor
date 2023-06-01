use crate::{Tensor, tensor::{TensorId}};

use super::{TensorCache, Var, Graph, Tape, gradient::backprop_graph, var::VarId, Tensors};

pub struct _Loss<Out:Tensors<Out=Out>>(Tensor, Out);

pub struct Trainer<In: Tensors, Out: Tensors> {
    _vars: Vec<(VarId, TensorId)>,
    fun: Box<dyn Fn(&Graph, In, &TensorCache) -> (Out, TensorCache)>,

    graph: Graph,
    gradients: Vec<(VarId, Graph)>,
}

pub struct Train<'a, In: Tensors<Out=In>, Out: Tensors<Out=Out>> {
    module: &'a Trainer<In, Out>,
    tensors: TensorCache,
    out: Out,
}

impl<In, Out> Trainer<In, Out>
where
    In: Tensors<Out=In>,
    Out: Tensors<Out=Out>,
{
    pub fn compile<F>(input: In, fun: F) -> Trainer<In, Out>
    where
        F: FnOnce(In::In<'_>) -> Out,
    {
        let mut tape = Tape::build(input, fun);

        let out_ids = tape.out_ids().clone();
        
        let mut backprop_graphs : Vec<(VarId, Graph)> = Vec::new();

        for var in tape.tracked_vars() {
            let id = tape.graph().get_id_by_var(var.id());

            let graph = backprop_graph(&tape.graph(), id);

            backprop_graphs.push((var.id(), graph));
        } 

        Self {
            _vars: Default::default(),
            fun: Box::new(move |graph: &Graph, input, fwd_tensors| { 
                let mut out = graph.tensors().clone();

                In::set_arg(&mut out, 0, &input);

                graph.apply(&mut out, fwd_tensors);

                let mut index = 0;
                let value = Out::make_out(&out, &out_ids, &mut index);

                (value, out)
            }),

            graph: tape.take_graph().unwrap(),
            gradients: backprop_graphs,
        }
    }

    pub fn call(&self, input: In) -> Out {
        let tensors = self.graph.tensors().clone();

        let (value, _out_tensors) = (self.fun)(&self.graph, input, &tensors);

        value
    }

    pub fn train(&self, input: In) -> Train<In, Out> {
        let (out, tensors) = (self.fun)(&self.graph, input, &self.graph.tensors());

        Train {
            module: self,
            out,
            tensors,
        }
    }

    pub(crate) fn get_var(&self, id: VarId) -> &Var {
        self.graph.get_var(id)
    }
}

impl<In:Tensors<Out=In>,Out:Tensors<Out=Out>> Train<'_, In, Out> {
    pub fn value(&self) -> Out {
        self.out.clone()
    }

    pub fn gradient(&self, var :&Var) -> Tensor {
        for (grad_var, grad_graph) in &self.module.gradients {
            if &var.id() == grad_var {
                let mut out = grad_graph.tensors().clone();

                grad_graph.apply(&mut out, &self.tensors);
        
                return out.last()
            }
        }

        panic!("{:?} is an unknown gradient", var);
    }

    pub(crate) fn gradients(&self) -> Vec<(VarId, Tensor)> {
        let mut vec = Vec::new();

        for (id, grad_graph) in &self.module.gradients {
            let mut out = grad_graph.tensors().clone();

            grad_graph.apply(&mut out, &self.tensors);
        
            vec.push((*id, out.last()))
        }

        vec
    }

}

#[cfg(test)]
mod test {
    use log::LevelFilter;

    use crate::model::{Tape};

    use crate::tensor::TensorId;
    use crate::{
        model::{Trainer, Var},
        tensor, Tensor,
    };

    #[test]
    fn test_alloc() {
        assert_eq!(Tape::alloc_id(), None);

        let _trainer = Trainer::compile((), |()| {
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

        let m_a = Trainer::compile((), 
            |_| a.tensor().clone()
        );

        let value = m_a.call(());
        assert_eq!(value, tensor!([[1.]]));

        let m_x = Trainer::compile((), 
            |_| x.tensor().clone()
        );

        let value = m_x.call(());
        assert_eq!(value, tensor!([2.]));
        // let train = module.train(());
    }

    #[test]
    fn binop_mul() {
        let a = Var::new("a", tensor!([1., 2., 3.]));

        let m_a = Trainer::compile((), 
        |_| &a * tensor!(2.)
        );

        let value = m_a.call(());
        assert_eq!(value, tensor!([2., 4., 6.]));
    }

    #[test]
    fn single_arg() {
        env_logger::builder().filter_level(LevelFilter::Debug).init();

        let a = Var::new("a", tensor!([1., 2., 3.]));

        let m_a = Trainer::compile(
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

        let m_a = Trainer::compile(
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

        let m_a = Trainer::compile(
            (tensor!([1., 1.]), tensor!([1., 1.])),
            |(x, y)| (y.clone(), x.clone())
        );

        let (y, x) = m_a.call((tensor!([2., 1.]), tensor!([1., 2.])));
        assert_eq!(x, tensor!([2., 1.]));
        assert_eq!(y, tensor!([1., 2.]));
    }

    #[test]
    fn grad_sub() {
        //env_logger::builder().filter_level(LevelFilter::Debug).init();

        let a = Var::new("a", tensor!([1., 2., 3.]));

        let m_a = Trainer::compile((), 
        |_| tensor!(2.) - &a
        ); // .training(&[&a]);

        let value = m_a.call(());
        assert_eq!(value, tensor!([1., 0., -1.]));

        let train = m_a.train(());

        assert_eq!(train.value(), tensor!([1., 0., -1.]));
        assert_eq!(train.gradient(&a), tensor!([-1., -1., -1.]));
    }

    #[test]
    fn grad_mul() {
        //env_logger::builder().filter_level(LevelFilter::Debug).init();

        let a = Var::new("a", tensor!([1., 2., 3.]));

        let m_a = Trainer::compile((), 
        |_| tensor!(2.) * &a
        ); // .training(&[&a]);

        let value = m_a.call(());
        assert_eq!(value, tensor!([2., 4., 6.]));

        let train = m_a.train(());

        assert_eq!(train.value(), tensor!([2., 4., 6.]));
        assert_eq!(train.gradient(&a), tensor!([2., 2., 2.]));
    }

    #[test]
    fn test_var() {
        let x = Var::new("x", tensor!(0.));

        let trainer = Trainer::compile((), |()| {
            x.tensor().clone()
        }); // .training(&[&x]);
        let train = trainer.train(());

        assert_eq!(train.gradient(&x), tensor!(1.));

        let x = Var::new("x", tensor!([[0., 2.], [10., 11.]]));
        let trainer = Trainer::compile((), |()| {
            x.tensor().clone()
        }); // .training(&[&x]);
        let train = trainer.train(());

        assert_eq!(train.gradient(&x), tensor!([[1., 1.], [1., 1.]]));
    }

    #[test]
    fn test_l2_loss() {
        let a = Var::new("a", tensor!(2.));

        let trainer = Trainer::compile((), |()| {
            a.l2_loss()
        }); // .training(&[&a]);
        let train = trainer.train(());
        let da = train.gradient(&a);
        assert_eq!(da, tensor!(2.0));

        let a = Var::new("a", tensor!(3.));
        let y = Var::new("y", tensor!(0.));

        let trainer = Trainer::compile((), |()| {
            let loss: Tensor = (&a - &y).l2_loss();
            assert_eq!(loss, tensor!(4.5));
            loss
        }); // .training(&[&a, &y]);
        let train = trainer.train(());

        let da = train.gradient(&a);
        assert_eq!(da, tensor!(3.0));

        let dy = train.gradient(&y);
        assert_eq!(dy, tensor!(-3.0));

        let a = Var::new("a", tensor!([1., 2.]));

        let trainer = Trainer::compile((), |()| {
            let loss: Tensor = a.l2_loss();

            assert_eq!(&loss, &tensor!(1.25));

            loss
        }); // .training(&[&a]);
        let train = trainer.train(());

        let da = train.gradient(&a);
        assert_eq!(da, tensor!([1.0, 2.0]));
    }

    #[test]
    fn test_sq_rank_0() {
        let a = Var::new("a", tensor!(1.));
        let x = Var::new("x", tensor!(0.));

        let trainer = Trainer::compile((), |()| {
            let loss: Tensor = (&a - &x) * (&a - &x);

            loss
        }); // .training(&[&a, &x]);
        let train = trainer.train(());

        assert_eq!(train.gradient(&a), tensor!(2.));
        assert_eq!(train.gradient(&x), tensor!(-2.));
    }

    #[test]
    fn test_sq_rank_1() {
        let a = Var::new("a", tensor!([1., 2.]));
        let x = Var::new("x", tensor!([0., 0.]));

        let trainer = Trainer::compile((), |()| {
            let loss: Tensor = (&a - &x) * (&a - &x);

            loss
        }); // .training(&[&a, &x]);
        let train = trainer.train(());

        assert_eq!(train.gradient(&a), tensor!(2.));
        assert_eq!(train.gradient(&x), tensor!(-2.));
    }

    #[test]
    fn test_sq_rank_2() {
        let a = Var::new("a", tensor!([[1., 2.], [3., 4.]]));
        let x = Var::new("x", tensor!([[0., 1.], [0., 2.]]));

        let trainer = Trainer::compile((), |()| {
            let loss: Tensor = (&a - &x) * (&a - &x);

            loss
        }); // .training(&[&a, &x]);
        let train = trainer.train(());

        assert_eq!(train.gradient(&a), tensor!(2.));
        assert_eq!(train.gradient(&x), tensor!(-2.));

        let trainer = Trainer::compile((), |()| {
            (&a - &x) * (&a - &x)
        }); // .training(&[&a, &x]);
        let train = trainer.train(());

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
