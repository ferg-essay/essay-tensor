use std::rc::Rc;

use crate::tensor::{Tensor, TensorData};


pub trait Uop {
    fn eval(&self, value: f32) -> f32;
}

pub trait Binop {
    fn eval(&self, a: f32, b: f32) -> f32;
}

pub fn uop<const N:usize>(uop: impl Uop, tensor: &Tensor<N>) -> Tensor<N> {
    let buffer = tensor.buffer();
    let len = buffer.len();

    unsafe {
        let mut data = TensorData::<f32>::new_uninit(len);

        for i in 0..len {
            data.uset(i, uop.eval(buffer.uget(i)));
        }

        Tensor::<N>::new(Rc::new(data), tensor.shape().clone())
    }
}

pub fn binop<const N:usize>(op: impl Binop, a: &Tensor<N>, b: &Tensor<N>) -> Tensor<N> {
    assert_eq!(a.shape(), b.shape());

    let a_data = a.buffer();
    let b_data = b.buffer();

    let len = a_data.len();

    unsafe {
        let mut data = TensorData::<f32>::new_uninit(len);

        for i in 0..len {
            data.uset(i, op.eval(
                a_data.uget(i), 
                b_data.uget(i)
            ));
        }

        Tensor::<N>::new(Rc::new(data), a.shape().clone())
    }
}

impl<F> Uop for F
where F: Fn(f32) -> f32 {
    fn eval(&self, value: f32) -> f32 {
        (self)(value)
    }
}

impl<F> Binop for F
where F: Fn(f32, f32) -> f32 {
    fn eval(&self, a: f32, b: f32) -> f32 {
        (self)(a, b)
    }
}
