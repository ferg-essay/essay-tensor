use super::{Var, Expr, TensorCache, Tensors, Operation, model::ModelContext};
use crate::{tensor::{TensorId}, Tensor};
use std::{cell::RefCell, sync::atomic::{AtomicU32, Ordering}};

pub struct Function<In: Tensors, Out: Tensors> {
    fun: Box<dyn Fn(&Expr, In) -> (Out, TensorCache)>,

    expr: Expr,

    vars: Vec<Var>,
}

impl<In, Out> Function<In, Out>
where
    In: Tensors,
    Out: Tensors,
{
    pub fn new<F>(input: In, fun: F) -> Function<In, Out>
    where
        F: FnMut(In, &mut ModelContext) -> Out,
    {
        let id = ModelId::alloc();

        let mut expr = Expr::new(id);
        let mut tensors = TensorCache::new(id);

        let mut ctx = ModelContext;
    
        let input = In::fun_in(&mut expr, &mut tensors, &input);

        let Tape {
            expr,
            tensors: _tensors,
            vars,
            out_ids,
        } = Tape::build(expr, tensors, input, fun, &mut ctx);

        Self {
            fun: Box::new(move |expr: &Expr, input| { 
                let mut out = expr.tensors().clone();

                In::set_input(&mut out, 0, &input);

                expr.call(&mut out, expr.tensors());

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
        let (value, _out) = (self.fun)(&self.expr, input);

        value
    }

    pub(crate) fn train(&self, input: In) -> (Out, TensorCache) {
        (self.fun)(&self.expr, input)
    }
}

pub struct Tape {
    expr: Expr,

    tensors: TensorCache,

    out_ids: Vec<TensorId>,

    vars: Vec<Var>,
}

impl Tape {
    fn build<F, In, Out>(
        expr: Expr, 
        tensors: TensorCache, 
        input: In, 
        fun: F,
        ctx: &mut ModelContext,
    ) -> Tape
    where
        In: Tensors,
        Out: Tensors,
        F: FnOnce(In, &mut ModelContext) -> Out,
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

        let out = fun(input, ctx);

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

static MODEL_ID: AtomicU32 = AtomicU32::new(0);

///
/// VarId is globally unique to avoid name collisions.
///
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ModelId(u32);

impl ModelId {
    pub(crate) fn alloc() -> ModelId {
        let id = MODEL_ID.fetch_add(1, Ordering::SeqCst);

        ModelId(id)
    }

    #[inline]
    pub fn index(&self) -> usize {
        self.0 as usize
    }
}

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
            |_, _| a.tensor().clone()
        );

        let value = m_a.call(());
        assert_eq!(value, tensor!([[1.]]));

        let m_x = Function::new((), 
            |_, _| x.tensor().clone()
        );

        let value = m_x.call(());
        assert_eq!(value, tensor!([2.]));
        // let train = module.train(());
    }

    #[test]
    fn binop_mul() {
        let a = Var::new("a", tensor!([1., 2., 3.]));

        let m_a = Function::new((), 
        |_, _| &a * tensor!(2.)
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
            |x, _| &a * x
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
            |(x, y), _| x - y
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
            |(x, y), _| (y.clone(), x.clone())
        );

        let (y, x) = m_a.call((tensor!([2., 1.]), tensor!([1., 2.])));
        assert_eq!(x, tensor!([2., 1.]));
        assert_eq!(y, tensor!([1., 2.]));
    }
}
