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

impl Tensor {
    pub fn uop(&self, uop: impl Uop<f32>) -> Self {
        let buffer = self.buffer();
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

    pub fn binop(&self, b: &Self, op: impl Binop<f32>) -> Self {
        let size = self.broadcast(b);
    
        let a_data = self.buffer();
        let b_data = b.buffer();
    
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
            self.next_binop(&b, data.init(), shape, op.to_op())
        }
    }

    pub fn fold(
        &self, 
        init: f32, 
        op: impl Fold<f32>, 
    ) -> Tensor<f32> {
        let a_data = self.buffer();
    
        let len = a_data.len();
        let stride = self.dim(0);
        let batch = len / stride;
    
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
    
            let shape = self.shape();
            let o_shape: Vec<usize> = shape[1..].iter().map(|d| *d).collect();

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
    
        let a_data = self.buffer();
        let b_data = b.buffer();

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
}

impl<F, D:Dtype> Uop<D> for F
where F: Fn(D) -> D + Clone + fmt::Debug + Sync + Send + 'static {
    fn eval(&self, value: D) -> D {
        (self)(value)
    }

    fn to_op(&self) -> Box<dyn Op> {
        Box::new(FnOp)
    }
}

impl<F, D:Dtype> Binop<D> for F
where F: Fn(D, D) -> D {
    fn eval(&self, a: D, b: D) -> D {
        (self)(a, b)
    }

    fn to_op(&self) -> Box<dyn Op> {
        Box::new(FnOp)
    }
}

#[derive(Debug)]
struct FnOp;

impl Op for FnOp {
    fn box_clone(&self) -> super::BoxOp {
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
