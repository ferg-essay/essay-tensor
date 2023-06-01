use super::{Var, Expr, TensorCache, Tensors, model::ModelId, Operation};
use crate::{tensor::{TensorId}, Tensor};
use std::cell::RefCell;

pub struct Function<In: Tensors, Out: Tensors> {
    fun: Box<dyn Fn(&Expr, In, &TensorCache) -> (Out, TensorCache)>,

    expr: Expr,

    vars: Vec<Var>,
}

impl<In, Out> Function<In, Out>
where
    In: Tensors<Item=In>,
    Out: Tensors<Item=Out>,
{
    pub fn new<F>(input: In, fun: F) -> Function<In, Out>
    where
        F: FnOnce(In::Item) -> Out,
    {
        let id = ModelId::alloc();

        let mut expr = Expr::new(id);
        let mut tensors = TensorCache::new(id);
    
        let input = In::fun_in(&mut expr, &mut tensors, &input);

        let Tape {
            expr,
            tensors: _tensors,
            vars,
            out_ids,
        } = Tape::build(expr, tensors, input, fun);

        Self {
            fun: Box::new(move |expr: &Expr, input, fwd_tensors| { 
                let mut out = expr.tensors().clone();

                In::set_arg(&mut out, 0, &input);

                expr.call(&mut out, fwd_tensors);

                let mut index = 0;
                let value = Out::make_out(&out, &out_ids, &mut index);

                (value, out)
            }),

            expr,
            vars,
        }
    }

    pub(crate) fn expr(&self) -> &Expr {
        &self.expr
    }

    pub fn vars(&self) -> &Vec<Var> {
        &self.vars
    }

    pub fn call(&self, input: In) -> Out {
        let tensors = self.expr.tensors().clone();

        let (value, out) = (self.fun)(&self.expr, input, &tensors);

        value
    }

    pub(crate) fn train(&self, input: In) -> (Out, TensorCache) {
        let tensors = self.expr.tensors().clone();

        (self.fun)(&self.expr, input, &tensors)
    }
}

pub struct Tape {
    expr: Expr,

    tensors: TensorCache,

    out_ids: Vec<TensorId>,

    vars: Vec<Var>,
}

impl Tape {
    fn build<F, In, Out>(expr: Expr, tensors: TensorCache, input: In, fun: F) -> Tape
    where
        In: Tensors<Item=In>,
        Out: Tensors<Item=Out>,
        F: FnOnce(In::Item) -> Out,
    {
        let tape = Tape {
            expr,
            tensors,
            out_ids: Default::default(),
            vars: Default::default(),
        };

        // TODO: add RAII guard?
        TAPE.with(|f| {
            assert!(f.borrow().is_none());
            f.borrow_mut().replace(tape);
        });

        let out = fun(input);

        let mut tape = TAPE.with(|f| f.borrow_mut().take().unwrap());

        Out::out_ids(&mut tape.out_ids, &out);

        tape
    }

    pub(crate) fn is_active() -> bool {
        TAPE.with(|f| f.borrow().is_some())
    }

    pub(crate) fn set_tensor(tensor: Tensor) -> Tensor {
        if tensor.id().is_some() {
            TAPE.with(|f| {
                if let Some(tape) = f.borrow_mut().as_mut() {
                    tape.expr.set_tensor(tensor.id(), tensor.clone());
                }
            })
        }

        tensor
    }

    pub(crate) fn var(var: &Var) -> Tensor {
        TAPE.with(|f| {
            if let Some(tape) = f.borrow_mut().as_mut() {
                tape.vars.push(var.clone());
                tape.expr.var(var)
            } else {
                var.tensor_raw()
            }
        })
    }

    pub(crate) fn constant(tensor: &Tensor) -> Tensor {
        TAPE.with(|f| {
            if let Some(tape) = f.borrow_mut().as_mut() {
                tape.expr.constant(tensor.clone())
            } else {
                tensor.clone()
            }
        })
    }

    pub(crate) fn op(op: Box<dyn Operation>, node_args: Vec<TensorId>) -> TensorId {
        TAPE.with(|f| {
            if let Some(tape) = f.borrow_mut().as_mut() {
                tape.expr.op(op, node_args)
            } else {
                TensorId::NONE
            }
        })
    }
}

thread_local! {
    pub static TAPE: RefCell<Option<Tape>>  = RefCell::new(None);
}

#[derive(Debug)]
pub enum TapeError {}

#[cfg(test)]
mod test {
    // use log::LevelFilter;

    use crate::{
        model::{function::Function, Var},
        tensor,
        Tensor,
    };

    #[test]
    fn var() {
        let a = Var::new("a", tensor!([[1.]]));
        let x = Var::new("x", tensor!([2.]));

        let m_a = Function::new((), 
            |_| a.tensor().clone()
        );

        let value = m_a.call(());
        assert_eq!(value, tensor!([[1.]]));

        let m_x = Function::new((), 
            |_| x.tensor().clone()
        );

        let value = m_x.call(());
        assert_eq!(value, tensor!([2.]));
        // let train = module.train(());
    }

    #[test]
    fn binop_mul() {
        let a = Var::new("a", tensor!([1., 2., 3.]));

        let m_a = Function::new((), 
        |_| &a * tensor!(2.)
        );

        let value = m_a.call(());
        assert_eq!(value, tensor!([2., 4., 6.]));
    }

    #[test]
    fn single_arg() {
        // env_logger::builder().filter_level(LevelFilter::Debug).init();

        let a = Var::new("a", tensor!([1., 2., 3.]));

        let m_a = Function::new(
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

        let m_a = Function::new(
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

        let m_a = Function::new(
            (tensor!([1., 1.]), tensor!([1., 1.])),
            |(x, y)| (y.clone(), x.clone())
        );

        let (y, x) = m_a.call((tensor!([2., 1.]), tensor!([1., 2.])));
        assert_eq!(x, tensor!([2., 1.]));
        assert_eq!(y, tensor!([1., 2.]));
    }
}
