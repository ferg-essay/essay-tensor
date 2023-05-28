use core::fmt;
use std::{any::type_name, marker::PhantomData};

use crate::{
    Tensor, 
    tensor::{Dtype, TensorId, TensorUninit}, 
    function::{NodeOp, Tape, Operation, IntoForward, Graph, graph::GradientOp}, prelude::Shape
};

pub trait ReduceKernel<S: State, D: Dtype=f32> : Clone + Copy + Send + Sync + 'static {
    fn f(&self, state: S, a: D) -> S;

    fn df_dx(&self, a: D) -> D;
}

pub fn reduce_op<Op, S>(a: impl Into<Tensor>, op: Op, opt: impl ReduceOpt) -> Tensor
where
    Op: ReduceKernel<S>,
    S: State<Value=f32>,
{
    let a = a.into();

    let reduce_op = ReduceCpu {
        op: op.clone(), 
        options: opt.into(),
        marker: PhantomData,
    };

    let node = NodeOp::new(&[&a], reduce_op.to_op());

    let tensor = reduce_op.forward(&[&a], node);

    Tape::set_tensor(tensor)
}

#[derive(PartialEq)]
pub struct ReduceCpu<Op: ReduceKernel<S>, S: State> {
    op: Op, 
    options: ReduceArg,
    marker: PhantomData<S>,
}

impl<Op: ReduceKernel<S>, S: State> ReduceCpu<Op, S> {
    #[inline]
    fn op(&self) -> Op {
        self.op
    }

    #[inline]
    fn axis(&self) -> Option<i32> {
        self.options.axis
    }

    fn output_shape(&self, shape: &Shape) -> (Shape, usize, usize, usize) {
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

    }
}

impl<Op, S> Operation for ReduceCpu<Op, S> 
where
    Op: ReduceKernel<S>,
    S: State<Value=f32>
{
    fn name(&self) -> &str {
        type_name::<Op>()
    }

    fn forward(
        &self,
        args: &[&Tensor],
        node: TensorId,
    ) -> Tensor {
        assert!(args.len() == 1);

        let a = args[0];

        let (o_shape, batch, a_len, inner) = self.output_shape(a.shape());

        unsafe {
            let mut o_data = TensorUninit::<f32>::new(o_shape.size());

            let a_ptr = a.as_ptr();
            let o_ptr = o_data.as_mut_ptr();

            let op = self.op();
    
            for n in 0..batch {
                for i in 0..inner {
                    let a_ptr = a_ptr.add(n * a_len * inner + i);

                    let mut state = S::default();

                    for k in 0..a_len {
                        state = op.f(
                            state, 
                            *a_ptr.add(k * inner), 
                        );
                    }

                    *o_ptr.add(n * inner + i) = state.value();
                }
            }

            Tensor::from_uninit_with_id(o_data, o_shape, node)
        }
    }

    fn back(
        &self,
        _forward: &Graph,
        graph: &mut Graph,
        i: usize,
        args: &[TensorId],
        prev: TensorId,
    ) -> TensorId {
        assert!(i == 0);

        graph.add_grad_op(self.clone(), &[args[0]], prev)
    }
}

impl<Op: ReduceKernel<S>, S: State> GradientOp for ReduceCpu<Op, S> {
    fn name(&self) -> &str {
        type_name::<Op>()
    }

    fn df(
        &self,
        args: &[&Tensor],
        prev: &Tensor,
    ) -> Tensor {
        let a = args[0];

        let (o_shape, batch, a_len, inner) = self.output_shape(a.shape());
        
        let len = a.len();
        
        unsafe {
            let out = TensorUninit::<f32>::new(len);

            let op = &self.op;

            for n in 0..batch {
                let a_ptr = a.as_ptr().add(n * a_len * inner);
                let o_ptr = out.as_ptr().add(n * a_len * inner);
                let prev_ptr = prev.as_ptr().add(n * inner);

                for i in 0..inner {
                    let a_ptr = a_ptr.add(i);
                    let o_ptr = o_ptr.add(i);
                    let prev_ptr = prev_ptr.add(i);
        
                    for k in 0..a_len {
                        let df_dx = op.df_dx(*a_ptr.add(k * inner));
                        let prev_df = *prev_ptr.add(i);

                        *o_ptr.add(k * inner) = df_dx * prev_df;
                    }
                }
            }
    
            Tensor::from_uninit(out, a.shape())
        }
    }
}

impl<Op: ReduceKernel<S>, S: State> Clone for ReduceCpu<Op, S> {
    fn clone(&self) -> Self {
        Self { 
            op: self.op.clone(), 
            options: self.options.clone(), 
            marker: self.marker.clone() 
        }
    }
}

pub trait ReduceOpt {
    fn axis(self, axis: Option<i32>) -> ReduceArg;
    fn into(self) -> ReduceArg;
}

#[derive(Default, Clone, Debug, PartialEq)]
pub struct ReduceArg {
    axis: Option<i32>,
}

impl ReduceOpt for ReduceArg {
    fn axis(self, axis: Option<i32>) -> ReduceArg {
        Self { axis, ..self }
    }

    fn into(self) -> ReduceArg {
        self
    }
}

impl ReduceOpt for () {
    fn axis(self, axis: Option<i32>) -> ReduceArg {
        ReduceArg::default().axis(axis)
    }

    fn into(self) -> ReduceArg {
        ReduceArg::default()
    }
}

pub trait State : Default + Send + Sync + 'static {
    type Value;

    fn value(&self) -> Self::Value;
}

impl<T: Dtype + Clone + Default + fmt::Debug> State for T {
    type Value = Self;

    fn value(&self) -> Self::Value {
        self.clone()
    }
}

#[cfg(test)]
mod test {
    use crate::{prelude::*, function::Var};

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

        let module = Trainer::compile((), |()| {
            2. * &x.l2_loss()
        });
        let train = module.train(());

        assert_eq!(train.value(), tf32!(10.));
        assert_eq!(train.gradient(&x), tf32!([2., 4., 4., 2.]));
    }
}
