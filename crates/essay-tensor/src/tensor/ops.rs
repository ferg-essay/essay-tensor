use crate::{Tensor};
use core::fmt;
use std::rc::Rc;

use crate::tensor::{TensorData, Dtype};

use super::{BoxOp, Op};


pub trait Uop<D:Dtype> : fmt::Debug + Sync + Send + 'static {
    fn eval(&self, value: D) -> D;

    //fn box_clone(&self) -> Box<dyn Uop<D>>;

    fn to_op(&self) -> Box<dyn Op>;
}

pub trait Binop<D:Dtype> {
    fn eval(&self, a: D, b: D) -> D;
}

#[derive(Debug)]
pub struct UopRecord<D:Dtype> {
    prev: Option<BoxOp>,
    uop: Box<dyn Uop<D>>,
}

#[derive(Debug)]
pub struct OpGraph {
    args: Vec<Option<OpGraph>>,
    //tensor: Option<Tensor<D>>,
    op: BoxOp,
}

impl OpGraph {
    pub fn new(args: &[&Option<OpGraph>], op: Box<dyn Op>) -> OpGraph {
        Self {
            args: args.iter().map(|g| 
                if let Some(graph) = g {
                    Some(graph.clone())
                } else {
                    None
                }
            ).collect(),
            op,
        }
    }
}

impl Clone for OpGraph {
    fn clone(&self) -> Self {
        Self { 
            args: self.args.clone(), 
            op: self.op.box_clone()
         }
    }
}
/*
impl<D:Dtype> UopRecord<D> {
    fn new(prev: &Option<BoxOp>, uop: Box<dyn Uop<D>>) -> UopRecord<D> {
        Self {
            prev: if let Some(op) = prev {
                Some(op.box_clone())
            } else {
                None
            },
            uop,
        }
    }
}
impl<D:Dtype> Op for UopRecord<D> {
    fn box_clone(&self) -> BoxOp {
        Box::new(self.clone())
    }
}

impl<D:Dtype> Clone for UopRecord<D> {
    fn clone(&self) -> Self {
        Self { 
            prev: if let Some(prev) = &self.prev {
                Some(prev.box_clone())
            } else {
                None
            },
            uop: self.uop.box_clone(),
        }
    }
}
*/
impl<const N:usize, D:Dtype> Tensor<N, D> {
    pub fn uop(self, uop: impl Uop<D>) -> Self {
        let buffer = self.buffer();
        let len = buffer.len();
    
        unsafe {
            let mut data = TensorData::<D>::new_uninit(len);
    
            for i in 0..len {
                data.uset(i, uop.eval(buffer.uget(i)));
            }
    
            Self::new_op(
                Rc::new(data), 
                self.shape().clone(),
                OpGraph::new(&[self.op()], uop.to_op()),
            )
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
where F: Fn(D) -> D + Clone + fmt::Debug + Sync + Send + 'static {
    fn eval(&self, value: D) -> D {
        (self)(value)
    }
    /*
    fn box_clone(&self) -> Box<dyn Uop<D>> {
        Box::new(self.clone())
    }
 */
    fn to_op(&self) -> Box<dyn Op> {
        todo!()
    }
}

impl<F, D:Dtype> Binop<D> for F
where F: Fn(D, D) -> D {
    fn eval(&self, a: D, b: D) -> D {
        (self)(a, b)
    }
}
