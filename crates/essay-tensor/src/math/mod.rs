use std::{ops};

use crate::{tensor::{Tensor, Uop, Binop, Op, BoxOp}};

#[derive(Debug, Clone)]
enum Unary {
    Abs,
    Cos,
    Exp,
    Ln,
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
            Unary::Neg => -value,
            Unary::Sin => value.sin(),
        }
    }

    fn to_op(&self) -> Box<dyn Op> {
        self.box_clone()
    }
}

macro_rules! tensor_uop {
    ($fun:ident, $op:expr) => {
        pub fn $fun(a: &Tensor) -> Tensor {
            a.uop($op)
        }

        impl Tensor {
            pub fn $fun(&self) -> Tensor {
                self.uop($op)
            }
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

impl Op for Unary {
    fn box_clone(&self) -> BoxOp {
        Box::new(self.clone())
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

    fn to_op(&self) -> Box<dyn Op> {
        self.box_clone()
    }
}

impl Op for Binary {
    fn gradient(&self, i: usize, args: &[&Tensor], prev: &Option<Tensor>) -> Tensor {
        match &self {
            Binary::Mul => {
                match prev {
                    Some(prev) => {
                        if i == 0 {
                            args[1] * prev
                        } else {
                            args[0] * prev
                        }
                    }
                    None => {
                        if i == 0 {
                            args[1].clone()
                        } else {
                            args[0].clone()
                        }
                    }
                }
            },
            Binary::Sub => {
                match prev {
                    Some(prev) => {
                        if i == 0 { 
                            prev.clone()
                        } else { 
                            - prev
                        }
                    }
                    None => {
                        if i == 0 { 
                            Tensor::ones(args[0].shape())
                        } else { 
                            Tensor::fill(-1., args[1].shape())
                        }
                    }
                }
            },
            Binary::Add => {
                match prev {
                    Some(prev) => prev.clone(),
                    None => Tensor::ones(args[0].shape())
                }
            },
            _ => todo!("{:?}", self)
        }
    }

    fn box_clone(&self) -> BoxOp {
        Box::new(self.clone())
    }
}

macro_rules! tensor_binop {
    ($fun:ident, $op:expr) => {
        pub fn $fun(a: &Tensor, b: &Tensor) -> Tensor {
            a.binop(b, $op)
        }

        impl Tensor {
            pub fn $fun(&self, b: &Tensor) -> Tensor {
                self.binop(b, $op)
            }
        }
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
