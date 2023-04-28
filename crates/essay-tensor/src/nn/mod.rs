use crate::{tensor::{Tensor, Uop, Op, BoxOp}};

#[derive(Debug, Clone)]
enum Unary {
    ReLU,
    Softplus,
}

impl Uop<f32> for Unary {
    fn eval(&self, value: f32) -> f32 {
        match &self {
            Unary::ReLU => value.max(0.),
            Unary::Softplus => (value.exp() + 1.).ln(),
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

impl<const N:usize> Tensor<N> {
    pub fn relu(self) -> Self {
        self.uop(Unary::ReLU)
    }

    pub fn softplus(self) -> Self {
        self.uop(Unary::Softplus)
    }
}
