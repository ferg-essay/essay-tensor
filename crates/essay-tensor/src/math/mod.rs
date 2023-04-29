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

impl Tensor {
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

impl Tensor {
    pub fn max(self, rhs: Self) -> Self {
        self.binop(Binary::Max, rhs)
    }
    
    pub fn min(self, rhs: Self) -> Self {
        self.binop(Binary::Min, rhs)
    }
}

impl ops::Add for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Self) -> Self::Output {
        self.binop(Binary::Add, rhs)
    }
}

impl ops::Sub for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Self) -> Self::Output {
        self.binop(Binary::Sub, rhs)
    }
}

impl ops::Mul<Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Self) -> Self::Output {
        self.binop(Binary::Mul, rhs)
    }
}
