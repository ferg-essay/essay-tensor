use std::{sync::Arc, any::type_name};

use crate::{module::{Graph, TensorId, ForwardOp, IntoForward, NodeOp, ModuleTape}, Tensor};

use super::{Dtype, TensorUninit, TensorData, NodeId};

pub trait Binop<D:Dtype=f32> : Clone + Send + Sync + 'static {
    fn eval(&self, a: D, b: D) -> D;

    fn backprop(
        &self,
        forward: &Graph,
        graph: &mut Graph,
        i: usize,
        args: &[TensorId],
        prev: TensorId,
    ) -> TensorId;
 
    // fn to_op(&self) -> Box<dyn ForwardOp>;
}

#[derive(Debug, Clone)]
pub struct BinopImpl<Op:Binop> {
    op: Op,
}

impl Tensor {
    pub fn binop<Op:Binop<f32>>(&self, b: &Self, op: Op) -> Self {
        let (shape, data) = self.eval_binop_impl(b, &op);

        let binop = BinopImpl::new(op);

        self.next_binop(&b, data, shape, binop)
    }

    pub fn eval_binop<Op:Binop<f32>>(
        &self, 
        b: &Self, 
        op: &Op
    ) -> Tensor {
        let (shape, data) = self.eval_binop_impl(b, op);

        Tensor::new(Arc::new(data), &shape)
    }

    pub fn eval_binop_impl<Op:Binop<f32>>(
        &self, 
        b: &Self, 
        op: &Op
    ) -> (Vec<usize>, TensorData) {
        let size = self.broadcast(b);
    
        let a_data = self.data();
        let b_data = b.data();
    
        unsafe {
            let mut data = TensorUninit::<f32>::new(size);
    
            for i in 0..size {
                data.set_unchecked(i, op.eval(
                    a_data[i],
                    b_data[i],
                ));
            }
    
            //Self::new(Rc::new(data), b.shape().clone())
    
            let shape = if self.rank() < b.rank() { 
                b.shape().clone() 
            } else { 
                self.shape().clone() 
            };

            (shape, data.init())
        }
    }

    pub fn next_binop(
        &self, 
        b: &Tensor,
        data: TensorData, 
        shape: Vec<usize>,
        into_op: impl IntoForward, // Box<dyn ForwardOp>
    ) -> Tensor {
        let tensor = Self::new_op(
            Arc::new(data), 
            shape, 
            NodeOp::new(&[self, b], into_op.to_op()),
        );

        if let NodeId::Id(id) = tensor.node() {
            ModuleTape::set_tensor(*id, tensor.clone());
        }

        tensor
    }
}


impl<Op:Binop<f32>> BinopImpl<Op> {
    fn new(op: Op) -> Self {
        Self {
            op,
        }
    }
}

impl<Op:Binop<f32>> ForwardOp for BinopImpl<Op> {
    fn name(&self) -> &str {
        type_name::<Op>()
    }
    
    fn eval(
        &self,
        _tensors: &crate::module::TensorCache,
        args: &[&Tensor],
    ) -> Tensor {
        // todo!("unimplemented op {}", self.name())
        args[0].eval_binop(args[1], &self.op)
    }

    fn backprop(
        &self,
        forward: &Graph,
        graph: &mut Graph,
        i: usize,
        args: &[TensorId],
        prev: TensorId,
    ) -> TensorId {
        self.op.backprop(forward, graph, i, args, prev)
    }
}

// TODO: debug seems wrong
impl<F, D:Dtype> Binop<D> for F
where F: Fn(D, D) -> D + Send + Sync + 'static + Clone {
    fn eval(&self, a: D, b: D) -> D {
        (self)(a, b)
    }
/*
    fn to_op(&self) -> Box<dyn ForwardOp> {
        Box::new(FnOp)
    }
*/
    fn backprop(
        &self,
        _forward: &Graph,
        _graph: &mut Graph,
        _i: usize,
        _args: &[TensorId],
        _prev: TensorId,
    ) -> TensorId {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::{*};

    #[test]
    fn binop_broadcast() {
        let a = tensor!([1., 2., 3.]);
        let b = tensor!(1.);

        assert_eq!(
            a.binop(&b, |a, b| 100. * a + b),
            tensor!([101., 201., 301.])
        );

        assert_eq!(
            b.binop(&a, |a, b| 100. * a + b),
            tensor!([101., 102., 103.])
        );

        let a = tensor!([1., 2.]);
        let b = tensor!([[1., 2.], [3., 4.]]);

        assert_eq!(
            a.binop(&b, |a, b| 100. * a + b),
            tensor!([[101., 202.], [103., 204.]])
        );

        assert_eq!(
            b.binop(&a, |a, b| 100. * a + b),
            tensor!([[101., 202.], [301., 402.]])
        );

        let a = tensor!([1., 2.]);
        let b = tensor!([
            [[1., 2.], [3., 4.]],
            [[10., 20.], [30., 40.]],
        ]);

        assert_eq!(
            a.binop(&b, |a, b| 100. * a + b),
            tensor!([
                [[101., 202.], [103., 204.]],
                [[110., 220.], [130., 240.]],
            ])
        );

        assert_eq!(
            b.binop(&a, |a, b| 100. * a + b),
            tensor!([
                [[101., 202.], [301., 402.]],
                [[1001., 2002.], [3001., 4002.]],
            ]),
        );
    }
}
