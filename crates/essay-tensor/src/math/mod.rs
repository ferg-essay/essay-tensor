use std::{rc::Rc, cmp};

use crate::{tensor::{Tensor, TensorData}, ops::{Uop, Binop, uop, binop}};

enum Unary {
    Abs,
    Cos,
    Exp,
    Ln,
    Sin,
}

impl Uop for Unary {
    fn eval(&self, value: f32) -> f32 {
        match &self {
            Unary::Abs => value.abs(),
            Unary::Cos => value.cos(),
            Unary::Exp => value.exp(),
            Unary::Ln => value.ln(),
            Unary::Sin => value.sin(),
        }
    }
}

enum Binary {
    Add,
    Max,
    Min,
    Sub,
}

impl Binop for Binary {
    fn eval(&self, a: f32, b: f32) -> f32 {
        match &self {
            Binary::Add => a + b,
            Binary::Max => a.max(b),
            Binary::Min => a.min(b),
            Binary::Sub => a - b,
        }
    }
}

pub fn abs<const N:usize>(tensor: &Tensor<N>) -> Tensor<N> {
    uop(Unary::Abs, tensor)
}

pub fn cos<const N:usize>(tensor: &Tensor<N>) -> Tensor<N> {
    uop(Unary::Cos, tensor)
}

pub fn exp<const N:usize>(tensor: &Tensor<N>) -> Tensor<N> {
    uop(Unary::Exp, tensor)
}

pub fn ln<const N:usize>(tensor: &Tensor<N>) -> Tensor<N> {
    uop(Unary::Ln, tensor)
}

pub fn sin<const N:usize>(tensor: &Tensor<N>) -> Tensor<N> {
    uop(Unary::Sin, tensor)
}

pub fn add<const N:usize>(a: &Tensor<N>, b: &Tensor<N>) -> Tensor<N> {
    binop(Binary::Add, a, b)
}

pub fn max<const N:usize>(a: &Tensor<N>, b: &Tensor<N>) -> Tensor<N> {
    binop(Binary::Max, a, b)
}

pub fn min<const N:usize>(a: &Tensor<N>, b: &Tensor<N>) -> Tensor<N> {
    binop(Binary::Min, a, b)
}

pub fn subtract<const N:usize>(a: &Tensor<N>, b: &Tensor<N>) -> Tensor<N> {
    binop(Binary::Sub, a, b)
}
