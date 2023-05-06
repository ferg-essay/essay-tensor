use crate::{tensor::{Tensor, Uop, BiFold, Fold}, model::{BackOp, TensorId, ForwardOp, BoxForwardOp, Graph}, math};

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
    L2Loss(f32),
}

impl Uop<f32> for Unary {
    fn eval(&self, value: f32) -> f32 {
        match &self {
            Unary::ReLU => value.max(0.),
            Unary::Softplus => (value.exp() + 1.).ln(),
        }
    }

    fn to_op(&self) -> Box<dyn ForwardOp> {
        self.box_clone()
    }
}

impl ForwardOp for Unary {
    fn box_clone(&self) -> BoxForwardOp {
        Box::new(self.clone())
    }

    fn backtrace_top(
        &self,
        forward: &Graph,
        graph: &mut Graph,
        i: usize,
        args: &[TensorId],
        tensor: TensorId,
    ) -> TensorId {
        todo!()
    }

    fn backtrace(
        &self,
        forward: &Graph,
        graph: &mut Graph,
        i: usize,
        args: &[TensorId],
        tensor: TensorId,
        prev: TensorId,
    ) -> TensorId {
        todo!()
    }

    fn eval(
        &self,
        tensors: &crate::model::TensorCache,
        args: &[&Tensor],
    ) -> Tensor {
        todo!()
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

impl Fold<f32> for UReduce {
    fn apply(&self, acc: f32, a: f32) -> f32 {
        match &self {
            UReduce::L2Loss(n_inv) => {
                acc + n_inv * a * a
            },
        }
    }

    fn to_op(&self) -> Box<dyn ForwardOp> {
        self.box_clone()
    }
}

impl ForwardOp for UReduce {
    fn backtrace(
        &self, 
        forward: &Graph,
        graph: &mut Graph,
        i: usize, 
        args: &[TensorId], 
        tensor: TensorId, 
        prev: TensorId
    ) -> TensorId {
        match self {
            UReduce::L2Loss(n) => {
                assert_eq!(i, 0, "{:?} reduce has only one argument", self);

                //Tensor::ones(args[0].shape())
                todo!()
            },
        }
    }

    fn backtrace_top(
        &self,
        forward: &Graph,
        graph: &mut Graph,
        i: usize,
        args: &[TensorId],
        tensor: TensorId,
    ) -> TensorId {
        match self {
            UReduce::L2Loss(n) => {
                assert_eq!(i, 0, "{:?} reduce has only one argument", self);

                //Tensor::ones(args[0].shape())
                //let id = graph.constant_id(tensor);
                //graph.add_op(math::Unary::Mul(2.), &[id])
                //args[0]
                graph.constant_id(args[0])
            },
        }
    }

    fn box_clone(&self) -> BoxForwardOp {
        Box::new(self.clone())
    }

    fn eval(
        &self,
        tensors: &crate::model::TensorCache,
        args: &[&Tensor],
    ) -> Tensor {
        todo!()
    }
}

impl BackOp for UReduce {
    fn gradient(&self, i: usize, args: &[&Tensor], prev: &Option<Tensor>) -> Tensor {
        match self {
            UReduce::L2Loss(n) => {
                match i {
                    0 => Tensor::ones(args[0].shape()),
                    _ => panic!("invalid argument")
                }
            },
        }
    }

    /*
    fn box_clone(&self) -> BoxForwardOp {
        //Box::new(self.clone())
        todo!()
    }
    */
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

    fn to_op(&self) -> Box<dyn ForwardOp> {
        //Box::new(self.clone())
        todo!()
    }
}

impl BackOp for BiReduce {
    fn gradient(&self, i: usize, args: &[&Tensor], prev: &Option<Tensor>) -> Tensor {
        match i {
            0 => args[0] - args[1],
            1 => args[1] - args[0],
            _ => panic!("invalid argument")
        }
    }
}

impl Tensor {
    pub fn l2_loss(&self) -> Tensor {
        let n = if self.rank() > 0 { self.dim(0) } else { 1 };
        let n_inv = 0.5 / n as f32;
        self.fold(0.0.into(), UReduce::L2Loss(n_inv))
    }

    pub fn x_l2_loss(&self, b: &Self) -> Tensor {
        let n = if self.rank() > 0 { self.dim(0) } else { 1 };
        let n_inv = 0.5 / n as f32;
        self.bi_fold(0.0.into(), BiReduce::L2Loss(n_inv), b)
    }
}
