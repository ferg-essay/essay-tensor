use crate::{tensor::{Tensor, Uop, Op, BoxOp, BiFold}};

#[derive(Debug, Clone)]
enum Unary {
    ReLU,
    Softplus,
}

#[derive(Debug, Clone)]
enum BiReduce {
    L2Loss(f32),
}

impl Uop<f32> for Unary {
    fn eval(&self, value: f32) -> f32 {
        match &self {
            Unary::ReLU => value.max(0.),
            Unary::Softplus => (value.exp() + 1.).ln(),
        }
    }

    fn to_op(&self) -> Box<dyn Op> {
        self.box_clone()
    }
}

impl Op for Unary {
    fn box_clone(&self) -> BoxOp {
        Box::new(self.clone())
    }
}

impl Tensor {
    pub fn relu(&self) -> Self {
        self.uop(Unary::ReLU)
    }

    pub fn softplus(&self) -> Self {
        self.uop(Unary::Softplus)
    }
}

impl BiFold<f32> for BiReduce {
    fn apply(&self, acc: f32, a: f32, b: f32) -> f32 {
        match &self {
            BiReduce::L2Loss(n_inv) => {
                let v = a - b;

                acc + n_inv * v * v
            },
        }
    }

    fn to_op(&self) -> Box<dyn Op> {
        self.box_clone()
    }
}

impl Op for BiReduce {
    fn gradient(&self, i: usize, args: &[&Tensor], next: Option<Tensor>) -> Tensor {
        match i {
            0 => args[0] - args[1],
            1 => args[1] - args[0],
            _ => panic!("invalid argument")
        }
    }

    fn box_clone(&self) -> BoxOp {
        Box::new(self.clone())
    }
}

impl Tensor {
    pub fn l2_loss(&self, b: &Self) -> Tensor {
        let n = if self.rank() > 0 { self.dim(0) } else { 1 };
        let n_inv = 0.5 / n as f32;
        self.bi_fold(0.0.into(), BiReduce::L2Loss(n_inv), b)
    }
}
