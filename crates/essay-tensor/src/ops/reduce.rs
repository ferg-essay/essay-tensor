use std::any::type_name;

use crate::{
    Tensor, 
    tensor::{Dtype, TensorId, TensorUninit, NodeId}, 
    function::{NodeOp, Tape, Operation, IntoForward, Graph, graph::BackOp}, prelude::Shape
};

pub trait ReduceKernel<D:Dtype=f32> : Clone + Copy + Send + Sync + 'static {
    fn f(&self, state: D, a: D) -> D;

    fn df_dx(&self, a: D) -> D;
}

pub fn reduce_op<Op>(a: &Tensor, op: Op, axis: Option<i32>) -> Tensor
where
    Op: ReduceKernel
{
    let fold_op = ReduceCpu(op.clone(), axis);

    let node = NodeOp::new(&[a], fold_op.to_op());

    let tensor = fold_op.forward(&[&a], node);

    Tape::set_tensor(tensor)
}

#[derive(Clone, Copy, PartialEq)]
pub struct ReduceCpu<Op: ReduceKernel>(Op, Option<i32>);

impl<Op: ReduceKernel> ReduceCpu<Op> {
    #[inline]
    fn op(&self) -> Op {
        self.0
    }

    #[inline]
    fn axis(&self) -> Option<i32> {
        self.1
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

impl<Op: ReduceKernel> Operation for ReduceCpu<Op> {
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
        let shape = a.shape();

        let (o_shape, batch, a_len, inner) = self.output_shape(a.shape());

        unsafe {
            let mut o_data = TensorUninit::<f32>::new(o_shape.size());

            let a_ptr = a.as_ptr();
            let o_ptr = o_data.as_mut_ptr();

            let op = self.op();
    
            for n in 0..batch {
                for i in 0..inner {
                    let a_ptr = a_ptr.add(n * a_len * inner + i);

                    let mut v = 0.0;

                    for k in 0..a_len {
                        v = op.f(
                            v, 
                            *a_ptr.add(k * inner), 
                        );
                    }

                    *o_ptr.add(n * inner + i) = v;
                }
            }

            Tensor::from_uninit_node(o_data, o_shape, node)
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

        graph.add_back_op(self.clone(), &[args[0]], prev)
    }
}

impl<Op: ReduceKernel> BackOp for ReduceCpu<Op> {
    fn name(&self) -> &str {
        type_name::<Op>()
    }

    fn df(
        &self,
        args: &[&Tensor],
        prev: &Tensor,
    ) -> Tensor {
        let a = &args[0];

        assert_eq!(a.len(), prev.len());
        
        let len = a.len();
        
        unsafe {
            let out = TensorUninit::<f32>::new(len);

            let op = &self.0;

            let a_ptr = a.as_ptr();
            let prev = prev.as_ptr();
            let o_ptr = out.as_ptr();
        
            for i in 0..len {
                let df_dx = op.df_dx(*a_ptr.add(i));
                let prev_df = *prev.add(i);

                *o_ptr.add(i) = df_dx * prev_df;
            }
    
            Tensor::from_uninit(out, a.shape())
        }
    }
}
