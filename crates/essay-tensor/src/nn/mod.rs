use crate::{tensor::{Tensor, Uop, Op, BoxOp, BiFold}};

#[derive(Debug, Clone)]
enum Unary {
    ReLU,
    Softplus,
}

#[derive(Debug, Clone)]
enum BiReduce {
    MSE,
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

impl<const N:usize> Tensor<N> {
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
            BiReduce::MSE => {
                let v = a - b;

                acc + v * v
            },
        }
    }

    fn to_op(&self) -> Box<dyn Op> {
        self.box_clone()
    }
}

impl Op for BiReduce {
    fn box_clone(&self) -> BoxOp {
        Box::new(self.clone())
    }
}

impl<const N:usize> Tensor<N> {
    pub fn mean_square_error<const M:usize>(&self, b: &Self) -> Tensor<M> {
        assert!(N == M + 1 || N == 0 && M == 0);

        let n = if N > 0 { self.shape()[0] } else { 1 };
        let n_f = n as f32;
        Tensor::<0>::from(1.0 / n_f) * self.bi_fold_impl(0.0.into(), BiReduce::MSE, b)
    }
}
