use crate::{Tensor, tensor::TensorUninit, model::{ForwardOp, Graph, TensorId, EvalOp, IntoForward}};

use crate::tensor::{Dtype};

pub trait Fold<D:Dtype=f32> : Clone {
    fn apply(&self, state: D, a: D) -> D;

    fn to_op(&self) -> Box<dyn ForwardOp>;
}

pub trait BiFold<D:Dtype=f32> : Clone {
    fn apply(&self, state: D, a: D, b: D) -> D;

    fn to_op(&self) -> Box<dyn ForwardOp>;
}

impl Tensor {
    pub fn fold<Op:Fold + IntoForward>(
        &self, 
        init: f32, 
        op: Op,
    ) -> Tensor<f32> {
        let a_data = self.data();

        let shape = self.shape();
        let o_shape: Vec<usize> = if shape.len() > 1 {
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
    
            self.next_uop(o_data.init(), o_shape, op)
        }
    }

    pub fn bi_fold<Op:BiFold + IntoForward>(
        &self, 
        init: f32, 
        op: Op,
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

            self.next_binop(&b, o_data.init(), o_shape, op)
        }
    }

    pub fn fold_1<Op:Fold + IntoForward>(
        &self, 
        init: f32, 
        op: Op,
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
            self.next_uop(o_data.init(), o_shape, op)
        }
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

impl EvalOp for FnOp {
    fn eval(
        &self,
        _tensors: &crate::model::TensorCache,
        _args: &[&Tensor],
    ) -> Tensor {
        todo!()
    }
}
