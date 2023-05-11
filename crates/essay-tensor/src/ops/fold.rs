use std::any::type_name;

use crate::{
    Tensor, 
    tensor::{Dtype, TensorId, TensorUninit, NodeId}, 
    graph::{NodeOp, Tape, Operation, IntoForward, Graph, graph::BackOp}
};

pub trait Fold<D:Dtype=f32> : Clone + Copy + Send + Sync + 'static {
    fn f(&self, state: D, a: D) -> D;

    fn df_dx(&self, a: D) -> D;
}

#[derive(Clone, Copy, PartialEq)]
pub struct FoldCpu<Op:Fold>(Op, f32);

pub fn fold_op<Op>(a: &Tensor, init: f32, op: Op) -> Tensor
where
    Op:Fold
{
    let fold_op = FoldCpu(op.clone(), init);

    let node = NodeOp::new(&[a], fold_op.to_op());

    let tensor = fold_op.forward(&[&a], node);

    Tape::set_tensor(tensor)
}

impl<Op:Fold> FoldCpu<Op> {
    #[inline]
    fn op(&self) -> Op {
        self.0
    }

    #[inline]
    fn init(&self) -> f32 {
        self.1
    }
}

impl<Op:Fold> Operation for FoldCpu<Op> {
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
        let o_shape: Vec<usize> = if shape.len() > 1 {
            shape[1..].iter().map(|d| *d).collect()
        } else {
            Vec::new()
        };

        let len = o_shape.iter().product();
        let inner_len = a.dim_zero();
    
        let o_data = unsafe {
            let mut o_data = TensorUninit::<f32>::new(len);

            let a_ptr = a_data.as_ptr();
            let o_ptr = o_data.as_mut_ptr();

            let op = self.op();
    
            for i in 0..len {
                let mut v = self.init();

                for j in 0..inner_len {
                    v = op.f(
                        v, 
                        *a_ptr.add(i * inner_len + j), 
                    );
                }

                *o_ptr.add(i) = v;
            }

            o_data.init()
        };
    
        Tensor::new_op(o_data, o_shape, node)
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

impl<Op:Fold> BackOp for FoldCpu<Op> {
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
        
            for i in 0..len {
                let df_dx = op.df_dx(a_data.get_unchecked(i));
                let prev_df = prev.get_unchecked(i);

                out.set_unchecked(i, df_dx * prev_df);
            }
    
            out.init()
        };
        
        let shape = a.shape().clone();
        Tensor::new(data, &shape)
    }
}
