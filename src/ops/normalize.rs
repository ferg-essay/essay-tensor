use core::fmt;
use std::any::type_name;

use crate::{
    Tensor, 
    tensor::{Dtype, TensorId, TensorUninit}, 
    model::{NodeOp, Tape, Operation, Expr, expr::{GradientOp, GradOperation}}, prelude::Shape
};

pub trait NormalizeKernel<D: Dtype=f32> : Clone + Copy + Send + Sync + 'static {
    type State;

    fn init(&self) -> Self::State;

    fn accum(&self, state: Self::State, a: D) -> Self::State;

    fn f(&self, state: &Self::State, a: D) -> D;

    fn df_dx(&self, a: D) -> D;
}

pub fn normalize_op<Op>(
    x: impl Into<Tensor>, 
    op: Op, 
    opt: impl NormalizeOpt
) -> Tensor
where
    Op: NormalizeKernel,
{
    let x: Tensor = x.into();

    let normalize_op = NormalizeCpu {
        op: op.clone(), 
        options: opt.into(),
    };

    let node = NodeOp::new(&[&x], Box::new(normalize_op.clone()));

    let tensor = normalize_op.f(&[&x], node);

    Tape::set_tensor(tensor)
}

#[derive(PartialEq)]
pub struct NormalizeCpu<Op: NormalizeKernel> {
    op: Op, 
    options: NormalizeArg,
}

impl<Op: NormalizeKernel> NormalizeCpu<Op> {
    #[inline]
    fn op(&self) -> Op {
        self.op
    }

    #[inline]
    fn _axis(&self) -> Option<i32> {
        self.options.axis
    }

    fn batch_shape(&self, shape: &Shape) -> (usize, usize) {
        (1, shape.size())
        /*
        match self.axis() {
            None => (Shape::scalar(), 1, shape.size(), 1),
            Some(axis) => {
                let slice = shape.as_slice();
                let rank = slice.len();
                let axis = ((axis + rank as i32) % rank as i32) as usize;
                assert!(axis < rank);

                if rank == 1 {
                    return (Shape::scalar(), 1, shape.size(), 1)
                }

                let mut vec = Vec::<usize>::new();

                let mut outer = 1;
                for v in &slice[0..axis] {
                    vec.push(*v);
                    outer *= *v;
                }
                
                let mut inner = 1;
                for v in &slice[axis + 1..] {
                    vec.push(*v);
                    inner *= *v;
                }

                (Shape::from(vec), outer, slice[axis], inner)
            }
        }
        */
    }
}

impl<Op> Operation<f32> for NormalizeCpu<Op> 
where
    Op: NormalizeKernel,
{
    fn name(&self) -> &str {
        type_name::<Op>()
    }

    fn f(
        &self,
        args: &[&Tensor],
        node: TensorId,
    ) -> Tensor {
        assert!(args.len() == 1);

        let x_arg = args[0];

        let (batch, inner) = self.batch_shape(x_arg.shape());

        unsafe {
            let mut o_data = TensorUninit::<f32>::new(x_arg.len());

            let x = x_arg.as_ptr();
            let o = o_data.as_mut_ptr();

            let op = self.op();
    
            for n in 0..batch {
                let x = x.add(n * inner);
                let o = o.add(n * inner);

                let mut state = op.init();

                for i in 0..inner {
                    state = op.accum(state, *x.add(i));
                }

                for i in 0..inner {
                    let value = op.f(&state, *x.add(i));

                    *o.add(i) = value;
                }
            }

            Tensor::from_uninit_with_id(o_data, x_arg.shape(), node)
        }
    }
}


impl<Op> GradOperation<f32> for NormalizeCpu<Op> 
where
    Op: NormalizeKernel,
{
    fn df(
        &self,
        _forward: &Expr,
        graph: &mut Expr,
        i: usize,
        args: &[TensorId],
        prev: TensorId,
    ) -> TensorId {
        assert!(i == 0);

        graph.add_grad_op(self.clone(), &[args[0]], prev)
    }
}

impl<Op: NormalizeKernel> GradientOp for NormalizeCpu<Op> {
    fn name(&self) -> &str {
        type_name::<Op>()
    }

    fn df(
        &self,
        _args: &[&Tensor],
        _prev: &Tensor,
    ) -> Tensor {
        todo!()
    }
}

impl<Op: NormalizeKernel> Clone for NormalizeCpu<Op> {
    fn clone(&self) -> Self {
        Self { 
            op: self.op.clone(), 
            options: self.options.clone(), 
        }
    }
}

pub trait NormalizeOpt {
    fn axis(self, axis: Option<i32>) -> NormalizeArg;
    fn into(self) -> NormalizeArg;
}

#[derive(Default, Clone, Debug, PartialEq)]
pub struct NormalizeArg {
    axis: Option<i32>,
}

impl NormalizeOpt for NormalizeArg {
    fn axis(self, axis: Option<i32>) -> NormalizeArg {
        Self { axis, ..self }
    }

    fn into(self) -> NormalizeArg {
        self
    }
}

impl NormalizeOpt for () {
    fn axis(self, axis: Option<i32>) -> NormalizeArg {
        NormalizeArg::default().axis(axis)
    }

    fn into(self) -> NormalizeArg {
        NormalizeArg::default()
    }
}

pub trait State : Default + Send + Sync + 'static {
    type Value;

    fn _value(&self) -> Self::Value;
}

impl<T: Dtype + Clone + Default + fmt::Debug> State for T {
    type Value = Self;

    fn _value(&self) -> Self::Value {
        self.clone()
    }
}

#[cfg(test)]
mod test {
    // use crate::{prelude::*, model::Var};
/*
    #[test]
    fn reduce_sum_n() {
        assert_eq!(tf32!([1.]).reduce_sum(), tf32!(1.));
        assert_eq!(tf32!([1., 10.]).reduce_sum(), tf32!(11.));
        assert_eq!(tf32!([10., 1.]).reduce_sum(), tf32!(11.));
    }

    #[test]
    fn reduce_sum_1xn() {
        assert_eq!(tf32!([[1.]]).reduce_sum(), tf32!([1.]));
        assert_eq!(tf32!([[1., 10.]]).reduce_sum(), tf32!([11.]));
        assert_eq!(tf32!([[10., 1.]]).reduce_sum(), tf32!([11.]));
    }

    #[test]
    fn reduce_sum_2xn() {
        assert_eq!(tf32!([[1.], [2.]]).reduce_sum(), tf32!([1., 2.]));
        assert_eq!(tf32!([[1., 10.], [2., 20.]]).reduce_sum(), tf32!([11., 22.]));
        assert_eq!(tf32!([[20., 2.], [10., 1.]]).reduce_sum(), tf32!([22., 11.]));
    }

    #[test]
    fn reduce_sum_2x1xn() {
        assert_eq!(tf32!([[[1.]], [[2.]]]).reduce_sum(), tf32!([[1.], [2.]]));
        assert_eq!(tf32!([[[1., 10.]], [[2., 20.]]]).reduce_sum(), tf32!([[11.], [22.]]));
        assert_eq!(tf32!([[[20., 2.]], [[10., 1.]]]).reduce_sum(), tf32!([[22.], [11.]]));
    }

    #[test]
    fn reduce_sum_1xn_axis_none() {
        assert_eq!(tf32!([[1.]]).reduce_sum_opt(()), tf32!(1.));
        assert_eq!(tf32!([[1.]]).reduce_sum_opt(().axis(None)), tf32!(1.));
        assert_eq!(tf32!([[1., 10.]]).reduce_sum_opt(().axis(None)), tf32!(11.));
        assert_eq!(tf32!([[10., 1.]]).reduce_sum_opt(().axis(None)), tf32!(11.));
    }

    #[test]
    fn reduce_sum_2xn_axis_none() {
        assert_eq!(tf32!([[1.], [2.]]).reduce_sum_opt(().axis(None)), tf32!(3.));
        assert_eq!(tf32!([[1., 10.], [100., 1000.]]).reduce_sum_opt(().axis(None)), tf32!(1111.));
    }

    #[test]
    fn reduce_sum_2x1x1xn_axis_none() {
        assert_eq!(tf32!([[[[1.]]], [[[2.]]]]).reduce_sum_opt(().axis(None)), tf32!(3.));
        assert_eq!(tf32!([[[[1., 10.]]], [[[100., 1000.]]]]).reduce_sum_opt(().axis(None)), tf32!(1111.));
    }

    #[test]
    fn reduce_sum_2xn_axis_0() {
        assert_eq!(tf32!([[1.], [2.]]).reduce_sum_opt(().axis(Some(0))), tf32!([3.]));
        assert_eq!(tf32!([[1., 10.], [100., 1000.]]).reduce_sum_opt(().axis(Some(0))), tf32!([101., 1010.]));
    }

    #[test]
    fn reduce_sum_2xn_axis_m1() {
        assert_eq!(tf32!([[1.], [2.]]).reduce_sum_opt(().axis(Some(-1))), tf32!([1., 2.]));
        assert_eq!(tf32!([[1., 10.], [100., 1000.]]).reduce_sum_opt(().axis(Some(-1))), tf32!([11., 1100.]));
    }

    #[test]
    fn l2_loss_df_n() {
        let x = Var::new("x", tf32!([1., 2., 2., 1.]));

        let module = Trainer::compile((), |(), _| {
            2. * &x.l2_loss()
        });
        let train = module.train(());

        assert_eq!(train.value(), tf32!(10.));
        assert_eq!(train.gradient(&x), tf32!([2., 4., 4., 2.]));
    }
    */
}
