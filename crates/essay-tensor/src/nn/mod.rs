use crate::{tensor::{Tensor, Uop}};

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

    fn box_clone(&self) -> Box<dyn Uop<f32>> {
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
