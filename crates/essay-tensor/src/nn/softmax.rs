use std::any::type_name;

use crate::{
    Tensor, 
    tensor::{Dtype, TensorUninit, NodeId}, 
    graph::{NodeOp, Tape, Operation, IntoForward, Graph, TensorId, graph::BackOp}
};

pub trait Softmax<D:Dtype=f32> : Clone + Copy + Send + Sync + 'static {
    fn weight(&self, a: D) -> D;

    fn f(&self, a: D, weight: D, inv_sum: D) -> D;

    fn df_dx(&self, a: D) -> D;
}

#[derive(Clone, Copy, PartialEq)]
pub struct SoftmaxCpu<Op:Softmax>(Op, usize);

#[derive(Clone, Copy, PartialEq)]
pub struct SoftmaxImpl;

pub fn softmax(a: &Tensor) -> Tensor {
    softmax_op(a, SoftmaxImpl, None)
}

impl Tensor {
    pub fn softmax(&self) -> Tensor {
        softmax(self)
    }
}

impl Softmax for SoftmaxImpl {
    fn weight(&self, a: f32) -> f32 {
        a.exp()
    }

    fn f(&self, _a: f32, weight: f32, inv_sum: f32) -> f32 {
        weight * inv_sum
    }

    fn df_dx(&self, a: f32) -> f32 {
        a.exp()
    }
}

pub fn softmax_op<Op>(a: &Tensor, op: Op, chunk: Option<usize>) -> Tensor
where
    Op:Softmax
{
    let chunk = match chunk {
        Some(chunk) => chunk,
        None => a.dim_zero(),
    };

    assert!(a.dim_zero() % chunk == 0);

    let softmax_op = SoftmaxCpu(op.clone(), chunk);

    let node = NodeOp::new(&[a], softmax_op.to_op());

    let tensor = softmax_op.forward(&[&a], node);

    Tape::set_tensor(tensor)
}

impl<Op:Softmax> SoftmaxCpu<Op> {
    #[inline]
    fn op(&self) -> Op {
        self.0
    }
}

impl<Op:Softmax> Operation for SoftmaxCpu<Op> {
    fn name(&self) -> &str {
        type_name::<Op>()
    }

    fn forward(
        &self,
        args: &[&Tensor],
        node: NodeId,
    ) -> Tensor {
        assert!(args.len() == 1);

        let a = args[0];
        let a_data = a.data();

        let shape = a.shape();

        let len = a.len();
        let inner_len = a.dim_zero();
        let batch = len / inner_len;

        let chunk = self.1;
        let n_chunks = inner_len / chunk;
    
        let o_data = unsafe {
            let mut o_data = TensorUninit::<f32>::new(len);

            let op = self.op();
    
            for batch in 0..batch {
                for n in 0..n_chunks {
                    let a_ptr = a_data.as_ptr().add(batch * inner_len + n * chunk);
                    let o_ptr = o_data.as_mut_ptr().add(batch * inner_len + n * chunk);
        
                    let mut sum = 0.;

                    for i in 0..chunk {
                        let v = op.weight(*a_ptr.add(i));

                        *o_ptr.add(i) = v;

                        sum += v;
                    }

                    let factor = if sum == 0. { 1. } else { 1. / sum };

                    // normalize
                    for i in 0..chunk {
                        let v = op.f(
                            *a_ptr.add(i),
                            *o_ptr.add(i), 
                            factor
                        );

                        *o_ptr.add(i) = v;
                    }
                }
            }

            o_data.init()
        };
    
        Tensor::new_op(o_data, shape.clone(), node)
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

        graph.add_back_op(self.clone(), &[args[0]], prev)
    }
}

impl<Op:Softmax> BackOp for SoftmaxCpu<Op> {
    fn name(&self) -> &str {
        type_name::<Op>()
    }

    fn df(
        &self,
        args: &[&Tensor],
        prev: &Tensor,
    ) -> Tensor {
        let a = &args[0];
        let a_data = a.data();
        let prev = prev.data();

        assert_eq!(a_data.len(), prev.len());
        
        let len = a_data.len();
        
        let data = unsafe {
            let mut out = TensorUninit::<f32>::new(len);

            let op = &self.0;

            let a_ptr = a.as_slice();
            let prev_ptr = prev.as_slice();
        
            for i in 0..len {
                let df_dx = op.df_dx(a_ptr[i]);
                let prev_df = prev_ptr[i];

                out[i] = df_dx * prev_df;
            }
    
            out.init()
        };
        
        let shape = a.shape().clone();
        Tensor::new(data, &shape)
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_softmax() {
        assert_eq!(tensor!(0.).softmax(), tensor!(1.));
        assert_eq!(tensor!(1.).softmax(), tensor!(1.));
        assert_eq!(tensor!([0., 1.]).softmax(), 
            tensor!([0.26894143, 0.7310586]));
        assert_eq!(tensor!([0., 1., 0., 0.]).softmax(), 
            tensor!([0.1748777, 0.47536686, 0.1748777, 0.1748777]));
        assert_eq!(tensor!([0., 10., 0., 0.]).softmax(), 
            tensor!([4.539375e-5, 0.99986386, 4.539375e-5, 4.539375e-5]));
    }
}