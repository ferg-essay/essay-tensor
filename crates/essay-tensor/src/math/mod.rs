use std::{ops};

use crate::{tensor::{Tensor, Uop, Binop}, model::{ForwardOp, TensorId, Graph, TensorCache, EvalOp}, tensor_uop, tensor_binop};

#[derive(Debug, Clone)]
pub enum Unary {
    Abs,
    Cos,
    Exp,
    Ln,
    Mul(f32),
    Neg,
    Sin,
}

impl Uop<f32> for Unary {
    fn eval(&self, value: f32) -> f32 {
        match &self {
            Unary::Abs => value.abs(),
            Unary::Cos => value.cos(),
            Unary::Exp => value.exp(),
            Unary::Ln => value.ln(),
            Unary::Mul(a) => a * value,
            Unary::Neg => -value,
            Unary::Sin => value.sin(),
        }
    }
}

tensor_uop!(abs, Unary::Abs);
tensor_uop!(cos, Unary::Cos);
tensor_uop!(exp, Unary::Exp);
tensor_uop!(ln, Unary::Ln);
tensor_uop!(sin, Unary::Sin);

pub fn neg(a: &Tensor) -> Tensor {
    a.uop(Unary::Neg)
}

impl ops::Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        self.uop(Unary::Neg)
    }
}

impl ops::Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        self.uop(Unary::Neg)
    }
}

impl EvalOp for Unary {
    fn eval(
        &self,
        _tensors: &TensorCache,
        args: &[&Tensor],
    ) -> Tensor {
        match self {
            Unary::Abs => args[0].abs(),
            Unary::Cos => args[0].cos(),
            Unary::Exp => args[0].exp(),
            Unary::Ln => args[0].ln(),
            Unary::Mul(a) => {
                (1. / *a) * args[0]
            },
            Unary::Neg => {
                neg(args[0])
            },
            Unary::Sin => args[0].sin(),
        }
    }
}

//
// binops
//

#[derive(Debug, Clone)]
enum Binary {
    Add,
    Atan2,
    Div,
    DivEuclid,
    Hypot,
    Log,
    Max,
    Min,
    Mul,
    Powf,
    Powi,
    Rem,
    RemEuclid,
    Sub,
}

impl Binop<f32> for Binary {
    #[inline]
    fn eval(&self, a: f32, b: f32) -> f32 {
        match &self {
            Binary::Add => a + b,
            Binary::Atan2 => a.atan2(b),
            Binary::Div => a / b,
            Binary::DivEuclid => a.div_euclid(b),
            Binary::Hypot => a.hypot(b),
            Binary::Log => a.log(b),
            Binary::Max => a.max(b),
            Binary::Min => a.min(b),
            Binary::Mul => a * b,
            Binary::Powf => a.powf(b),
            Binary::Powi => a.powi(b as i32),
            Binary::Rem => a % b,
            Binary::RemEuclid => a.rem_euclid(b),
            Binary::Sub => a - b,
        }
    }

    fn backprop(
        &self,
        _forward: &Graph,
        graph: &mut Graph,
        i: usize,
        _args: &[TensorId],
        prev: TensorId,
    ) -> TensorId {
        assert!(i <= 1);

        match self {
            Binary::Add => {
                prev
            },
            Binary::Sub => {
                match i {
                    0 => { prev },
                    1 => { graph.add_op(Unary::Neg, &[prev]) }
                    _ => { panic!() },
                }
            },
            _ => todo!("backtrace {:?}", self)
        }
    }

    fn to_op(&self) -> Box<dyn ForwardOp> {
        //self.box_clone()
        todo!()
    }
}

tensor_binop!(atan2, Binary::Atan2);
tensor_binop!(div_euclid, Binary::DivEuclid);
tensor_binop!(hypot, Binary::Hypot);
tensor_binop!(log, Binary::Log);
tensor_binop!(max, Binary::Max);
tensor_binop!(min, Binary::Min);
tensor_binop!(powf, Binary::Powf);
tensor_binop!(powi, Binary::Powi);
tensor_binop!(rem_euclid, Binary::RemEuclid);

//
// overloaded operations: Add, Sub, Mul
//

macro_rules! tensor_ops {
    ($op:ident, $fun:ident) => {
        pub fn $fun(a: &Tensor, b: &Tensor) -> Tensor {
            a.binop(b, Binary::$op)
        }

        impl ops::$op for &Tensor {
            type Output = Tensor;
        
            fn $fun(self, rhs: Self) -> Self::Output {
                self.binop(&rhs, Binary::$op)
            }
        }

        impl ops::$op for Tensor {
            type Output = Tensor;
        
            fn $fun(self, rhs: Self) -> Self::Output {
                self.binop(&rhs, Binary::$op)
            }
        }

        impl ops::$op<&Tensor> for Tensor {
            type Output = Tensor;
        
            fn $fun(self, rhs: &Tensor) -> Self::Output {
                self.binop(&rhs, Binary::$op)
            }
        }

        impl ops::$op<Tensor> for &Tensor {
            type Output = Tensor;
        
            fn $fun(self, rhs: Tensor) -> Self::Output {
                self.binop(&rhs, Binary::$op)
            }
        }

        impl ops::$op<&Tensor> for f32 {
            type Output = Tensor;
        
            fn $fun(self, rhs: &Tensor) -> Self::Output {
                Tensor::from(self).binop(&rhs, Binary::$op)
            }
        }

        impl ops::$op<Tensor> for f32 {
            type Output = Tensor;
        
            fn $fun(self, rhs: Tensor) -> Self::Output {
                Tensor::from(self).binop(&rhs, Binary::$op)
            }
        }
    }
}

tensor_ops!(Add, add);
tensor_ops!(Div, div);
tensor_ops!(Mul, mul);
tensor_ops!(Rem, rem);
tensor_ops!(Sub, sub);

impl ops::Mul<Option<Tensor>> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Option<Tensor>) -> Self::Output {
        match rhs {
            Some(rhs) => self * rhs,
            None => self,
        }
    }
}

impl ops::Add<Option<Tensor>> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: Option<Tensor>) -> Self::Output {
        match rhs {
            Some(rhs) => self + rhs,
            None => self.clone(),
        }
    }
}

impl ops::Add<Option<Tensor>> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Option<Tensor>) -> Self::Output {
        match rhs {
            Some(rhs) => self + rhs,
            None => self,
        }
    }
}

impl ops::Mul<Option<Tensor>> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Option<Tensor>) -> Self::Output {
        match rhs {
            Some(rhs) => self * rhs,
            None => self.clone(),
        }
    }
}
