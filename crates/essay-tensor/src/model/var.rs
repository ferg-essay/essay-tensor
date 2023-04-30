use core::fmt;

use crate::{tensor::{Dtype, Op, BoxOp}, Tensor, prelude::IntoTensor};

use super::OpGraph;

pub struct Var<D:Dtype=f32> {
    name: String,
    tensor: Tensor<D>,
}

#[derive(Debug, Clone)]
pub struct VarOp(String);

impl<D:Dtype> Var<D> {
    pub fn new(name: &str, tensor: Tensor<D>) -> Self {
        Self {
            tensor: tensor.set_op(OpGraph::new::<D>(&[], VarOp(name.to_string()).box_clone())),
            name: name.to_string(),
        }
    }

    pub fn tensor(&self) -> Tensor<D> {
        self.tensor.clone()
    }
}

impl<D:Dtype> Tensor<D> {
    pub fn as_var(self, name: &str) -> Var<D> {
        Var::new(name, self)
    }
}

impl fmt::Debug for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Var")
            .field("name", &self.name)
            .field("tensor", &self.tensor)
            .finish()
    }
}

impl Op for VarOp {
    fn box_clone(&self) -> BoxOp {
        Box::new(self.clone())
    }
}

impl<D:Dtype> IntoTensor<D> for Var<D> {
    fn into_tensor(&self) -> Tensor<D> {
        self.tensor.clone()
    }
}

impl<D:Dtype> From<Var<D>> for Tensor<D> {
    fn from(value: Var<D>) -> Self {
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
