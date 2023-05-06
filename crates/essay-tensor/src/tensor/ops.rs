use crate::{Tensor, tensor::TensorUninit, model::{ForwardOp, BoxForwardOp, Graph, TensorId}};
use core::fmt;
use std::marker::PhantomData;

use crate::tensor::{Dtype};

pub trait Uop<D:Dtype> : fmt::Debug + Sync + Send + 'static {
    fn eval(&self, value: D) -> D;

    //fn box_clone(&self) -> Box<dyn Uop<D>>;

    fn to_op(&self) -> Box<dyn ForwardOp>;
}

pub struct UopImpl;

pub trait Binop<D:Dtype> : Send + Sync + 'static {
    fn eval(&self, a: D, b: D) -> D;

    fn backtrace(
        &self,
        forward: &Graph,
        graph: &mut Graph,
        i: usize,
        args: &[TensorId],
        tensor: TensorId,
        prev: TensorId,
    ) -> TensorId;
 
    fn to_op(&self) -> Box<dyn ForwardOp>;
}

pub trait Fold<D:Dtype> {
    fn apply(&self, state: D, a: D) -> D;

    fn to_op(&self) -> Box<dyn ForwardOp>;
}

pub trait BiFold<D:Dtype> {
    fn apply(&self, state: D, a: D, b: D) -> D;

    fn to_op(&self) -> Box<dyn ForwardOp>;
}

#[derive(Debug)]
pub struct BinopImpl<Op:Binop<f32>> {
    op: Op,
}

impl<Op:Binop<f32>> ForwardOp for BinopImpl<Op> {
    fn box_clone(&self) -> BoxForwardOp {
        todo!()
    }

    fn backtrace_top(
        &self,
        forward: &Graph,
        graph: &mut Graph,
        i: usize,
        args: &[TensorId],
        tensor: TensorId,
    ) -> TensorId {
        todo!()
    }

    fn backtrace(
        &self,
        forward: &Graph,
        graph: &mut Graph,
        i: usize,
        args: &[TensorId],
        tensor: TensorId,
        prev: TensorId,
    ) -> TensorId {
        self.op.backtrace(forward, graph, i, args, tensor, prev)
    }

    fn eval(
        &self,
        tensors: &crate::model::TensorCache,
        args: &[&Tensor],
    ) -> Tensor {
        todo!()
    }
}

impl<Op:Binop<f32>> BinopImpl<Op> {
    pub fn to_op(op: Op) -> Box<dyn ForwardOp> {
        let binop: BinopImpl<Op> = BinopImpl {
            op: op,
        };

        Box::new(binop)
    }

    pub fn binop(a: &Tensor, b: &Tensor, op: Op) -> Tensor {
        let size = a.broadcast(b);
    
        let a_data = a.data();
        let b_data = b.data();
    
        unsafe {
            let mut data = TensorUninit::<f32>::new(size);
    
            for i in 0..size {
                data.set_unchecked(i, op.eval(
                    a_data[i],
                    b_data[i],
                ));
            }
    
            let shape = if a.rank() < b.rank() { 
                b.shape().clone() 
            } else { 
                a.shape().clone() 
            };

            a.next_binop(&b, data.init(), shape, op.to_op())
        }
    }

}

impl Tensor {
    pub fn uop(&self, uop: impl Uop<f32>) -> Self {
        let buffer = self.data();
        let len = buffer.len();

        unsafe {
            let mut data = TensorUninit::<f32>::new(len);
    
            for i in 0..len {
                data.set_unchecked(i, uop.eval(buffer.get_unchecked(i)));
            }
    
            let shape = self.shape().clone();
            self.next_uop(data.init(), Vec::from(shape), uop.to_op())
        }
    }

    pub fn binop<Op:Binop<f32>>(&self, b: &Self, op: Op) -> Self {
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
            self.next_binop(&b, data.init(), shape, 
                BinopImpl::<Op>::to_op(op))
        }
    }

    pub fn fold(
        &self, 
        init: f32, 
        op: impl Fold<f32>, 
    ) -> Tensor<f32> {
        let a_data = self.data();

        let shape = self.shape();
        let o_shape: Vec<usize> = if shape.len() > 0 {
            shape[1..].iter().map(|d| *d).collect()
        } else {
            Vec::new()
        };

        let len = o_shape.iter().product();
        let stride = self.dim_zero();
        let batch = self.len() / stride;
    
        unsafe {
            let mut o_data = TensorUninit::<f32>::new(len);
    
            for i in 0..batch {
                let offset = i * stride;

                let mut v = init;

                for j in 0..stride {
                    v = op.apply(
                        v, 
                        a_data.get_unchecked(offset + j), 
                    );
                }

                o_data.set_unchecked(i, v);
            }
    
            //Self::new(Rc::new(data), b.shape().clone())
    
            self.next_uop(o_data.init(), o_shape, op.to_op())
        }
    }

    pub fn bi_fold(
        &self, 
        init: f32, 
        op: impl BiFold<f32>, 
        b: &Self
    ) -> Tensor<f32> {
        assert_eq!(self.shape(), b.shape());
    
        let a_data = self.data();
        let b_data = b.data();

        let len = a_data.len();
        let stride = if self.rank() > 0 { self.dim(0) } else { 1 };
        let batch = len / stride;
    
        unsafe {
            let mut o_data = TensorUninit::<f32>::new(len);
    
            for i in 0..batch {
                let offset = i * stride;

                let mut state = init;

                for j in 0..stride {
                    state = op.apply(
                        state, 
                        a_data.get_unchecked(offset + j), 
                        b_data.get_unchecked(offset + j)
                    );
                }

                o_data.set_unchecked(i, state);
            }
    
            //Self::new(Rc::new(data), b.shape().clone())
    
            let shape = self.shape();
            let o_shape: Vec<usize> = if shape.len() > 0 {
                shape[1..].iter().map(|d| *d).collect()
            } else {
                Vec::new()
            };

            self.next_binop(&b, o_data.init(), o_shape, op.to_op())
        }
    }

    pub fn fold_1(
        &self, 
        init: f32, 
        op: impl Fold<f32>, 
    ) -> Tensor<f32> {
        assert!(self.rank() >= 2);

        let a_data = self.data();

        // let shape = self.shape();
        let mut o_shape: Vec<usize> = Vec::new();
        let dim_0 = self.dim(0);
        let dim_1 = self.dim(1);
        o_shape.push(dim_0);
        for i in 2..self.rank() {
            o_shape.push(i);
        }
        let len : usize = o_shape.iter().product();
        let batch_len : usize = o_shape[1..].iter().product();
    
        unsafe {
            let mut o_data = TensorUninit::<f32>::new(len);
    
            for batch in 0..batch_len {
                let a_start = batch * dim_0 * dim_1;
                let o_start = batch * dim_0;

                for i in 0..dim_0 {
                    let mut value = init;

                    for j in 0..dim_1 {
                        value = op.apply(value, a_data[a_start + j * dim_0 + i]);
                    }

                    o_data[o_start + i] = value;
                }
            }
    
            //Self::new(Rc::new(data), b.shape().clone())
            // TODO: fold has different op
            self.next_uop(o_data.init(), o_shape, op.to_op())
        }
    }
}

impl<F, D:Dtype> Uop<D> for F
where F: Fn(D) -> D + Clone + fmt::Debug + Sync + Send + 'static {
    fn eval(&self, value: D) -> D {
        (self)(value)
    }

    fn to_op(&self) -> Box<dyn ForwardOp> {
        Box::new(FnOp)
    }
}

// TODO: debug seems wrong
impl<F, D:Dtype> Binop<D> for F
where F: Fn(D, D) -> D + Send + Sync + 'static {
    fn eval(&self, a: D, b: D) -> D {
        (self)(a, b)
    }

    fn to_op(&self) -> Box<dyn ForwardOp> {
        Box::new(FnOp)
    }

    fn backtrace(
        &self,
        forward: &Graph,
        graph: &mut Graph,
        i: usize,
        args: &[TensorId],
        tensor: TensorId,
        prev: TensorId,
    ) -> TensorId {
        todo!()
    }
}

impl<F, D:Dtype> Fold<D> for F
where F: Fn(D, D) -> D + Send + Sync + Clone + 'static {
    fn apply(&self, state: D, a: D) -> D {
        self(state, a)
    }

    fn to_op(&self) -> Box<dyn ForwardOp> {
        // TODO: placeholder
        Box::new(FnOp)
    }
}

#[derive(Debug)]
struct FnOp;

impl ForwardOp for FnOp {
    fn box_clone(&self) -> BoxForwardOp {
        todo!()
    }

    fn backtrace_top(
        &self,
        forward: &Graph,
        graph: &mut Graph,
        i: usize,
        args: &[TensorId],
        tensor: TensorId,
    ) -> crate::model::TensorId {
        todo!()
    }

    fn backtrace(
        &self,
        forward: &Graph,
        graph: &mut Graph,
        i: usize,
        args: &[TensorId],
        tensor: TensorId,
        prev: TensorId,
    ) -> TensorId {
        todo!()
    }

    fn eval(
        &self,
        tensors: &crate::model::TensorCache,
        args: &[&Tensor],
    ) -> Tensor {
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
