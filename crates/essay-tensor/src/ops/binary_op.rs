use std::{any::type_name};

use crate::{eval::{Graph, Operation, IntoForward, NodeOp, Tape, graph::BackOp}, Tensor, 
    tensor::{Dtype, TensorId, TensorUninit, NodeId}
};

pub trait BinaryKernel<D:Dtype=f32> : Clone + Copy + Send + Sync + 'static {
    fn f(&self, x: D, y: D) -> D;

    fn df_dx(&self, x: D, y: D) -> D;
    fn df_dy(&self, x: D, y: D) -> D;
}

#[derive(Debug, Clone)]
pub struct BinopImpl<Op:BinaryKernel>(Op);

#[derive(Debug, Clone)]
pub struct BinopDx<Op:BinaryKernel>(Op);

#[derive(Debug, Clone)]
pub struct BinopDy<Op:BinaryKernel>(Op);

pub fn binary_op<Op:BinaryKernel<f32>>(a: &Tensor, b: &Tensor<f32>, op: Op) -> Tensor {
    let binop = BinopImpl(op.clone());

    let node = NodeOp::new(&[a, b], binop.to_op());

    let tensor = binop.forward(&[a, b], node);

    Tape::set_tensor(tensor)
}

impl<Op:BinaryKernel<f32>> Operation for BinopImpl<Op> {
    fn name(&self) -> &str {
        type_name::<Op>()
    }
    
    fn forward(
        &self,
        args: &[&Tensor],
        node: NodeId,
    ) -> Tensor {
        let a = args[0];
        let b = args[1];

        let size = a.broadcast(b);
        let inner = a.len().min(b.len());
        let batch = size / inner;

        let shape = if a.rank() < b.rank() { 
            b.shape().clone() 
        } else { 
            a.shape().clone() 
        };

        unsafe {
            let mut out = TensorUninit::<f32>::new(size);

            let op = self.0;

            let o_ptr = out.as_mut_ptr();

            for n in 0..batch {
                let a_ptr = a.as_wrap_ptr(n * inner);
                let b_ptr = b.as_wrap_ptr(n * inner);

                for i in 0..inner {
                    *o_ptr.add(n * inner + i) = op.f(
                        *a_ptr.add(i), 
                        *b_ptr.add(i)
                    );
                }
            }

            Tensor::from_uninit_node(out, shape, node)
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
        match i {
            0 => graph.add_back_op(BinopDx(self.0.clone()), &[args[0], args[1]], prev),
            1 => graph.add_back_op(BinopDy(self.0.clone()), &[args[0], args[1]], prev),
            _ => unimplemented!(),
        }
    }
}

impl<Op:BinaryKernel<f32>> BackOp for BinopDx<Op> {
    fn name(&self) -> &str {
        type_name::<Op>()
    }

    fn df(
        &self,
        args: &[&Tensor],
        prev: &Tensor,
    ) -> Tensor {
        let x = &args[0];
        let y = &args[1];
        let len = x.len();
        
        unsafe {
            let mut data = TensorUninit::<f32>::new(len);

            let x_ptr = x.as_ptr();
            let y_ptr = y.as_ptr();
            let prev = prev.as_ptr();

            let o_ptr = data.as_mut_ptr();

            let op = &self.0;
        
            for i in 0..len {
                let x = *x_ptr.add(i);
                let y = *y_ptr.add(i);

                let df_dx = op.df_dx(x, y);
                let prev_df = *prev.add(i);

                *o_ptr.add(i) = df_dx * prev_df;
            }
    
            Tensor::from_uninit(data, x.shape())
        }
    }
}

impl<Op:BinaryKernel<f32>> BackOp for BinopDy<Op> {
    fn name(&self) -> &str {
        type_name::<Op>()
    }

    fn df(
        &self,
        args: &[&Tensor],
        prev: &Tensor,
    ) -> Tensor {
        let x = &args[0];
        let y = &args[1];
        let len = x.len();
        
        unsafe {
            let mut out = TensorUninit::<f32>::new(len);

            let x_ptr = x.as_ptr();
            let y_ptr = y.as_ptr();
            let prev = prev.as_ptr();

            let o_ptr = out.as_mut_ptr();
    
            let op = &self.0;
        
            for i in 0..len {
                let x = *x_ptr.add(i);
                let y = *y_ptr.add(i);

                let df_dx = op.df_dy(x, y);
                let prev_df = *prev.add(i);

                *o_ptr.add(i) = df_dx * prev_df;
            }
    
            Tensor::from_uninit(out, x.shape())
        }
    }
}

// TODO: debug seems wrong
impl<F, D:Dtype> BinaryKernel<D> for F
where F: Fn(D, D) -> D + Send + Sync + 'static + Clone + Copy {
    fn f(&self, x: D, y: D) -> D {
        (self)(x, y)
    }

    fn df_dx(&self, _x: D, _y: D) -> D {
        todo!()
    }

    fn df_dy(&self, _x: D, _y: D) -> D {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use crate::{prelude::{*}, ops::binary_op};

    #[test]
    fn binop_broadcast() {
        let a = tensor!([1., 2., 3.]);
        let b = tensor!(1.);

        assert_eq!(
            binary_op(&a, &b, |a, b| 100. * a + b),
            tensor!([101., 201., 301.])
        );

        assert_eq!(
            binary_op(&b, &a, |a, b| 100. * a + b),
            tensor!([101., 102., 103.])
        );

        let a = tensor!([1., 2.]);
        let b = tensor!([[1., 2.], [3., 4.]]);

        assert_eq!(
            binary_op(&a, &b, |a, b| 100. * a + b),
            tensor!([[101., 202.], [103., 204.]])
        );

        assert_eq!(
            binary_op(&b, &a, |a, b| 100. * a + b),
            tensor!([[101., 202.], [301., 402.]])
        );

        let a = tensor!([1., 2.]);
        let b = tensor!([
            [[1., 2.], [3., 4.]],
            [[10., 20.], [30., 40.]],
        ]);

        assert_eq!(
            binary_op(&a, &b, |a, b| 100. * a + b),
            tensor!([
                [[101., 202.], [103., 204.]],
                [[110., 220.], [130., 240.]],
            ])
        );

        assert_eq!(
            binary_op(&b, &a, |a, b| 100. * a + b),
            tensor!([
                [[101., 202.], [301., 402.]],
                [[1001., 2002.], [3001., 4002.]],
            ]),
        );
    }
}
