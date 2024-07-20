use core::fmt;
use std::any::type_name;

use crate::{model::{NodeOp, Tape, Operation, Expr, expr::{GradientOp, GradOperation}}, Tensor, 
    tensor::{Dtype, TensorUninit, TensorId}, prelude::Shape
};

pub fn init_op<Op>(op: Op, shape: impl Into<Shape>) -> Tensor
where
    Op: InitKernel<f32>
{
    let shape = shape.into();

    let uop = InitCpu(op.clone(), shape);

    let id = NodeOp::new(&[], Box::new(uop.clone()));

    let tensor = uop.f(&[], id);

    Tape::set_tensor(tensor)
}

pub trait InitKernel<D:Dtype> : fmt::Debug + Clone + PartialEq + Sync + Send + 'static
{
    type State;

    fn init(&self, shape: &Shape) -> Self::State;

    fn f(&self, state: &mut Self::State) -> D;
}

#[derive(Clone, PartialEq)]
pub struct InitCpu<Op:InitKernel<f32>>(Op, Shape);

impl<Op: InitKernel<f32>> Operation<f32> for InitCpu<Op> {
    fn name(&self) -> &str {
        type_name::<Op>()
    }
    
    fn f(
        &self,
        _args: &[&Tensor],
        id: TensorId,
    ) -> Tensor {
        let shape = self.shape();
        let len = shape.size();
    
        unsafe {
            let mut out = TensorUninit::<f32>::new(len);
    
            let op = &self.0;
            let o_ptr = out.as_mut_ptr();

            let mut state = op.init(&shape);
        
            for i in 0..len {
                *o_ptr.add(i) = op.f(&mut state);
            }
    
            Tensor::from_uninit_with_id(out, shape, id)
        }
    }
}

impl<Op: InitKernel<f32>> GradOperation<f32> for InitCpu<Op> {
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

impl<Op: InitKernel<f32>> GradientOp for InitCpu<Op> {
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


impl<Op: InitKernel<f32>> InitCpu<Op> {
    fn shape(&self) -> &Shape {
        &self.1
    }
}
