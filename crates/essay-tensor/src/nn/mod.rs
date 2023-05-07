use crate::{tensor::{Tensor, Uop, Fold}, model::{TensorId, ForwardOp, Graph, EvalOp}, tensor_uop};

#[derive(Debug, Clone)]
enum Unary {
    ReLU,
    Softplus,
}

#[derive(Debug, Clone)]
enum UReduce {
    L2Loss(f32),
}

#[derive(Debug, Clone)]
enum BiReduce {
}

impl Uop<f32> for Unary {
    fn eval(&self, value: f32) -> f32 {
        match &self {
            Unary::ReLU => value.max(0.),
            Unary::Softplus => (value.exp() + 1.).ln(),
        }
    }

    fn to_op(&self) -> Box<dyn ForwardOp> {
        Box::new(self.clone())
    }
}

impl EvalOp for Unary {
    fn eval(
        &self,
        _tensors: &crate::model::TensorCache,
        _args: &[&Tensor],
    ) -> Tensor {
        todo!()
    }
}

tensor_uop!(relu, Unary::ReLU);
tensor_uop!(softplus, Unary::Softplus);

impl Tensor {
    pub fn l2_loss(&self) -> Tensor {
        let n = self.dim_zero();
        let n_inv = 0.5 / n as f32;
        self.fold(0.0.into(), UReduce::L2Loss(n_inv))
    }
}

impl Fold<f32> for UReduce {
    fn apply(&self, acc: f32, a: f32) -> f32 {
        match &self {
            UReduce::L2Loss(n_inv) => {
                acc + n_inv * a * a
            },
        }
    }

    fn to_op(&self) -> Box<dyn ForwardOp> {
        Box::new(self.clone())
    }
}

impl ForwardOp for UReduce {
    fn eval(
        &self,
        _tensors: &crate::model::TensorCache,
        _args: &[&Tensor],
    ) -> Tensor {
        todo!()
    }

    fn backprop(
        &self, 
        _forward: &Graph,
        graph: &mut Graph,
        i: usize, 
        args: &[TensorId], 
        _prev: TensorId
    ) -> TensorId {
        match self {
            UReduce::L2Loss(_) => {
                assert_eq!(i, 0, "{:?} reduce has only one argument", self);

                graph.constant_id(args[0])
            },
        }
    }
}
