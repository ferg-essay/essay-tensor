use crate::{Tensor, tensor::TensorUninit};
use core::fmt;

use crate::tensor::{Dtype};

use super::{Op};


pub trait Uop<D:Dtype> : fmt::Debug + Sync + Send + 'static {
    fn eval(&self, value: D) -> D;

    //fn box_clone(&self) -> Box<dyn Uop<D>>;

    fn to_op(&self) -> Box<dyn Op>;
}

pub trait Binop<D:Dtype> {
    fn eval(&self, a: D, b: D) -> D;

    fn to_op(&self) -> Box<dyn Op>;
}

pub trait Fold<D:Dtype> {
    fn apply(&self, state: D, a: D) -> D;

    fn to_op(&self) -> Box<dyn Op>;
}

pub trait BiFold<D:Dtype> {
    fn apply(&self, state: D, a: D, b: D) -> D;

    fn to_op(&self) -> Box<dyn Op>;
}

impl<const N:usize, D:Dtype> Tensor<N, D> {
    pub fn uop(&self, uop: impl Uop<D>) -> Self {
        let buffer = self.buffer();
        let len = buffer.len();
    
        unsafe {
            let mut data = TensorUninit::<D>::new(len);
    
            for i in 0..len {
                data.set_unchecked(i, uop.eval(buffer.get_unchecked(i)));
            }
    
            let shape = self.shape().clone();
            self.next_uop(data.init(), shape, uop.to_op())
        }
    }

    pub fn binop(&self, op: impl Binop<D>, b: Self) -> Self {
        assert_eq!(self.shape(), b.shape());
    
        let a_data = self.buffer();
        let b_data = b.buffer();
    
        let len = a_data.len();
    
        unsafe {
            let mut data = TensorUninit::<D>::new(len);
    
            for i in 0..len {
                data.set_unchecked(i, op.eval(
                    a_data.get_unchecked(i), 
                    b_data.get_unchecked(i)
                ));
            }
    
            //Self::new(Rc::new(data), b.shape().clone())
    
            let shape = b.shape().clone();
            self.next_binop(&b, data.init(), shape, op.to_op())
        }
    }

    fn fold_impl<const M:usize>(
        &self, 
        init: D, 
        op: impl Fold<D>, 
    ) -> Tensor<M, D> {
        assert_eq!(N, M + 1);
    
        let a_data = self.buffer();
    
        let len = a_data.len();
        let stride = self.shape()[0];
        let batch = len / stride;
    
        unsafe {
            let mut o_data = TensorUninit::<D>::new(len);
    
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
    
            let shape = self.shape();
            let mut o_shape = [0; M];
            for i in 1..shape.len() {
                o_shape[i] = shape[i];
            }

            self.next_uop(o_data.init(), o_shape, op.to_op())
        }
    }

    fn bi_fold_impl<const M:usize>(
        &self, 
        init: D, 
        op: impl BiFold<D>, 
        b: &Self
    ) -> Tensor<M, D> {
        assert_eq!(N, M + 1);
        assert_eq!(self.shape(), b.shape());
    
        let a_data = self.buffer();
        let b_data = b.buffer();

        let len = a_data.len();
        let stride = self.shape()[0];
        let batch = len / stride;
    
        unsafe {
            let mut o_data = TensorUninit::<D>::new(len);
    
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
            let mut o_shape = [0; M];
            for i in 1..shape.len() {
                o_shape[i] = shape[i];
            }
            self.next_binop(&b, o_data.init(), o_shape, op.to_op())
        }
    }
}

macro_rules! fold {
    ( $n:expr, $m:expr ) => {
        impl<D:Dtype> Tensor<$n, D> {
            pub fn fold(&self, init: D, op: impl Fold<D>) -> Tensor<$m, D> {
                self.fold_impl(init, op)
            }
        }

        impl<D:Dtype> Tensor<$n, D> {
            pub fn bi_fold(&self, init: D, op: impl BiFold<D>, b: &Self) -> Tensor<$m, D> {
                self.bi_fold_impl(init, op, b)
            }
        }
    }
}

fold!(1, 0);
fold!(2, 1);
fold!(3, 2);
fold!(4, 3);
fold!(5, 4);

impl<F, D:Dtype> Uop<D> for F
where F: Fn(D) -> D + Clone + fmt::Debug + Sync + Send + 'static {
    fn eval(&self, value: D) -> D {
        (self)(value)
    }

    fn to_op(&self) -> Box<dyn Op> {
        todo!()
    }
}

impl<F, D:Dtype> Binop<D> for F
where F: Fn(D, D) -> D {
    fn eval(&self, a: D, b: D) -> D {
        (self)(a, b)
    }

    fn to_op(&self) -> Box<dyn Op> {
        todo!()
    }
}
