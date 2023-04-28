use crate::{tensor::{Dtype, Op, BoxOp, OpGraph}, Tensor, prelude::IntoTensor};

pub struct Var<const N:usize, D:Dtype=f32> {
    _name: String,
    tensor: Tensor<N, D>,
}

#[derive(Debug, Clone)]
pub struct VarOp(String);

impl<const N:usize, D:Dtype> Var<N, D> {
    pub fn new(name: &str, tensor: Tensor<N, D>) -> Self {
        Self {
            tensor: tensor.set_op(OpGraph::new(&[], VarOp(name.to_string()).box_clone())),
            _name: name.to_string(),
        }
    }

    pub fn tensor(&self) -> Tensor<N, D> {
        self.tensor.clone()
    }
}

impl<const N:usize, D:Dtype> Tensor<N, D> {
    pub fn as_var(self, name: &str) -> Var<N, D> {
        Var::new(name, self)
    }
}

impl Op for VarOp {
    fn box_clone(&self) -> BoxOp {
        Box::new(self.clone())
    }
}

impl<const N:usize, D:Dtype> IntoTensor<N, D> for Var<N, D> {
    fn into_tensor(&self) -> Tensor<N, D> {
        self.tensor.clone()
    }
}

impl<const N:usize, D:Dtype> From<Var<N, D>> for Tensor<N,D> {
    fn from(value: Var<N,D>) -> Self {
        value.tensor.clone()
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_var() {
        let t1 = tensor!([1., 2., 3.]);
        let v1 = t1.as_var("t1");

        let t2 = v1.tensor().exp();
        println!("t2: {:#?}", t2);

        println!("t2: {:#?}", v1.tensor());

        let t3 = tensor!([1., 2., 3.]);
        let t3 = t3.exp();
        println!("t3: {:#?}", t3);
    }
}
