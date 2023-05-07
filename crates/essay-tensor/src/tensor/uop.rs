use core::fmt;

use crate::{model::{IntoForward}, Tensor};

use super::{Dtype, TensorUninit};


pub trait Uop<D:Dtype> : fmt::Debug + Clone + Sync + Send + 'static {
    fn eval(&self, value: D) -> D;
}

impl Tensor {
    pub fn uop<Op>(&self, uop: Op) -> Self
    where
        Op:Uop<f32> + IntoForward
    {
        let buffer = self.data();
        let len = buffer.len();

        unsafe {
            let mut data = TensorUninit::<f32>::new(len);
    
            for i in 0..len {
                data.set_unchecked(i, uop.eval(buffer.get_unchecked(i)));
            }
    
            let shape = self.shape().clone();
            self.next_uop(data.init(), Vec::from(shape), uop)
        }
    }
}

 
impl<F, D:Dtype> Uop<D> for F
where F: Fn(D) -> D + Clone + fmt::Debug + Sync + Send + 'static {
    fn eval(&self, value: D) -> D {
        (self)(value)
    }
}
