use crate::{Tensor};
use std::rc::Rc;

use crate::tensor::{TensorData, Dtype};


pub trait Uop<D:Dtype> {
    fn eval(&self, value: D) -> D;
}

pub trait Binop<D:Dtype> {
    fn eval(&self, a: D, b: D) -> D;
}

impl<const N:usize, D:Dtype> Tensor<N, D> {
    pub fn uop(self, uop: impl Uop<D>) -> Self {
        let buffer = self.buffer();
        let len = buffer.len();
    
        unsafe {
            let mut data = TensorData::<D>::new_uninit(len);
    
            for i in 0..len {
                data.uset(i, uop.eval(buffer.uget(i)));
            }
    
            Self::new(Rc::new(data), self.shape().clone())
        }
    }

    pub fn binop(self, op: impl Binop<D>, b: Self) -> Self {
        assert_eq!(self.shape(), b.shape());
    
        let a_data = self.buffer();
        let b_data = b.buffer();
    
        let len = a_data.len();
    
        unsafe {
            let mut data = TensorData::<D>::new_uninit(len);
    
            for i in 0..len {
                data.uset(i, op.eval(
                    a_data.uget(i), 
                    b_data.uget(i)
                ));
            }
    
            Self::new(Rc::new(data), b.shape().clone())
        }
    }
}

impl<F, D:Dtype> Uop<D> for F
where F: Fn(D) -> D {
    fn eval(&self, value: D) -> D {
        (self)(value)
    }
}

impl<F, D:Dtype> Binop<D> for F
where F: Fn(D, D) -> D {
    fn eval(&self, a: D, b: D) -> D {
        (self)(a, b)
    }
}
