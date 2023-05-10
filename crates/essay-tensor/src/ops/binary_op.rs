use std::{any::type_name};

use crate::{module::{Graph, TensorId, ForwardOp, IntoForward, NodeOp, Tape, graph::BackOp}, Tensor, 
    tensor::{Dtype, TensorUninit, NodeId}
};

pub trait Binop<D:Dtype=f32> : Clone + Copy + Send + Sync + 'static {
    fn f(&self, x: D, y: D) -> D;

    fn df_dx(&self, x: D, y: D) -> D;
    fn df_dy(&self, x: D, y: D) -> D;
}

#[derive(Debug, Clone)]
pub struct BinopImpl<Op:Binop>(Op);

#[derive(Debug, Clone)]
pub struct BinopDx<Op:Binop>(Op);

#[derive(Debug, Clone)]
pub struct BinopDy<Op:Binop>(Op);

pub fn binary_op<Op:Binop<f32>>(a: &Tensor, b: &Tensor, op: Op) -> Tensor {
    let binop = BinopImpl(op.clone());

    let node = NodeOp::new(&[a, b], binop.to_op());

    let tensor = binop.eval(&[a, b], node);

    Tape::set_tensor(tensor)
}

impl<Op:Binop<f32>> ForwardOp for BinopImpl<Op> {
    fn name(&self) -> &str {
        type_name::<Op>()
    }
    
    fn eval(
        &self,
        args: &[&Tensor],
        node: NodeId,
    ) -> Tensor {
        let a = args[0];
        let b = args[1];

        let size = a.broadcast(b);
    
        let a_data = a.data();
        let b_data = b.data();
    
        unsafe {
            let mut data = TensorUninit::<f32>::new(size);

            let op = self.0;

            let o_ptr = data.as_mut_ptr();

            for i in 0..size {
                let value = op.f(
                    a_data.read_wrap(i), 
                    b_data.read_wrap(i)
                );

                *o_ptr.add(i) = value;
            }
    
            let shape = if a.rank() < b.rank() { 
                b.shape().clone() 
            } else { 
                a.shape().clone() 
            };

            Tensor::new_op(data.init(), shape, node)
        }
    }

    fn backprop(
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

impl<Op:Binop<f32>> BackOp for BinopDx<Op> {
    fn name(&self) -> &str {
        type_name::<Op>()
    }

    fn df(
        &self,
        args: &[&Tensor],
        prev: &Tensor,
    ) -> Tensor {
        let x = &args[0];
        let x_ptr = x.data();
        let y = &args[1];
        let y_ptr = y.data();
        let prev = prev.data();
        let len = x_ptr.len();
        
        let data = unsafe {
            let mut data = TensorUninit::<f32>::new(len);

            let op = &self.0;
        
            for i in 0..len {
                let x = x_ptr.get_unchecked(i);
                let y = y_ptr.get_unchecked(i);

                let df_dx = op.df_dx(x, y);
                let prev_df = prev.get_unchecked(i);

                data.set_unchecked(i, df_dx * prev_df);
            }
    
            data.init()
        };
        
        let shape = x.shape().clone();
        Tensor::new(data, &shape)
    }
}

impl<Op:Binop<f32>> BackOp for BinopDy<Op> {
    fn name(&self) -> &str {
        type_name::<Op>()
    }

    fn df(
        &self,
        args: &[&Tensor],
        prev: &Tensor,
    ) -> Tensor {
        let x = &args[0];
        let x_ptr = x.data();
        let y = &args[1];
        let y_ptr = y.data();
        let prev = prev.data();
        let len = x_ptr.len();
        
        let data = unsafe {
            let mut data = TensorUninit::<f32>::new(len);

            let op = &self.0;
        
            for i in 0..len {
                let x = x_ptr.get_unchecked(i);
                let y = y_ptr.get_unchecked(i);

                let df_dx = op.df_dy(x, y);
                let prev_df = prev.get_unchecked(i);

                data.set_unchecked(i, df_dx * prev_df);
            }
    
            data.init()
        };
        
        let shape = x.shape().clone();
        Tensor::new(data, &shape)
    }
}

// TODO: debug seems wrong
impl<F, D:Dtype> Binop<D> for F
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
