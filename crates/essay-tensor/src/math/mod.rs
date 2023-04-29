use std::{ops, rc::Rc};

use crate::{tensor::{Tensor, Uop, Binop, Op, BoxOp, TensorUninit}};

#[derive(Debug, Clone)]
enum Unary {
    Abs,
    Cos,
    Exp,
    Ln,
    Sin,
}

impl Uop<f32> for Unary {
    fn eval(&self, value: f32) -> f32 {
        match &self {
            Unary::Abs => value.abs(),
            Unary::Cos => value.cos(),
            Unary::Exp => value.exp(),
            Unary::Ln => value.ln(),
            Unary::Sin => value.sin(),
        }
    }
/*
    fn box_clone(&self) -> Box<dyn Uop<f32>> {
        Box::new(self.clone())
    }
     */

    fn to_op(&self) -> Box<dyn Op> {
        self.box_clone()
    }
}

impl Op for Unary {
    fn box_clone(&self) -> BoxOp {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
enum Binary {
    Add,
    Max,
    Min,
    Mul,
    Sub,
}

impl Binop<f32> for Binary {
    fn eval(&self, a: f32, b: f32) -> f32 {
        match &self {
            Binary::Add => a + b,
            Binary::Max => a.max(b),
            Binary::Min => a.min(b),
            Binary::Mul => a * b,
            Binary::Sub => a - b,
        }
    }

    fn to_op(&self) -> Box<dyn Op> {
        self.box_clone()
    }
}

impl Op for Binary {
    fn box_clone(&self) -> BoxOp {
        Box::new(self.clone())
    }
}

impl<const N:usize> Tensor<N> {
    pub fn abs(&self) -> Self {
        self.uop(Unary::Abs)
    }

    pub fn cos(&self) -> Self {
        self.uop(Unary::Cos)
    }
    
    pub fn exp(&self) -> Self {
        self.uop(Unary::Exp)
    }
    
    pub fn ln(&self) -> Self {
        self.uop(Unary::Ln)
    }
    
    pub fn sin(&self) -> Self {
        self.uop(Unary::Sin)
    }
}

//
// binops
//

impl<const N:usize> Tensor<N> {
    pub fn max(self, rhs: Self) -> Self {
        self.binop(Binary::Max, rhs)
    }
    
    pub fn min(self, rhs: Self) -> Self {
        self.binop(Binary::Min, rhs)
    }
}

impl<const N:usize> ops::Add for Tensor<N> {
    type Output = Tensor<N>;

    fn add(self, rhs: Self) -> Self::Output {
        self.binop(Binary::Add, rhs)
    }
}

impl<const N:usize> ops::Sub for Tensor<N> {
    type Output = Tensor<N>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.binop(Binary::Sub, rhs)
    }
}

impl<const N:usize> ops::Mul<Tensor<N>> for Tensor<0> {
    type Output = Tensor<N>;

    fn mul(self, rhs: Tensor<N>) -> Self::Output {
        let len = rhs.len();
        let a = self.get(0).unwrap();

        unsafe {
            let b_data = rhs.buffer();
            let mut o_data = TensorUninit::new(len);

            for i in 0..len {
                o_data[i] = a * b_data[i];
            }

            // TODO: graph
            Tensor::new(Rc::new(o_data.init()), rhs.shape().clone())
        }
    }
}

impl<const N:usize> ops::Mul<Tensor<N>> for f32 {
    type Output = Tensor<N>;

    fn mul(self, rhs: Tensor<N>) -> Self::Output {
        Tensor::<0>::from(self).mul(rhs)
    }
}
