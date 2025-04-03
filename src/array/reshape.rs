
//
// reshape operation
//

use crate::{
    Tensor, 
    prelude::Shape, 
    tensor::Dtype, 
};

pub fn reshape<D: Dtype>(x: &Tensor<D>, shape: impl Into<Shape>) -> Tensor<D> {
    let shape = shape.into();
    let op = ReshapeOp(shape);

    //let node = NodeOp::new(&[x], Box::new(op.clone()));
    //let id = TensorId::unset();

    //let tensor = op.f(&[&x], id);

    //Tape::set_tensor(tensor)
    //tensor

    todo!();
}

impl<D: Dtype> Tensor<D> {
    pub fn reshape(&self, shape: impl Into<Shape>) -> Tensor<D> {
        reshape(self, shape)
    }
}

#[derive(Clone)]
pub struct ReshapeOp(Shape);
/*
impl<D: Dtype> Operation<D> for ReshapeOp {
    fn f(
        &self,
        args: &[&Tensor<D>],
        id: TensorId,
    ) -> Tensor<D> {
        let tensor = args[0];

        assert_eq!(tensor.shape().size(), self.0.size());

        tensor.reshape_impl(&self.0, id)
    }
}
    */

#[cfg(test)]
mod test {
    use crate::{prelude::*, array::{reshape}};
    
    #[test]
    fn test_reshape() {
        assert_eq!(
            reshape(&tf32!([[1., 2.], [3., 4.]]), [4]), 
            tf32!([1., 2., 3., 4.])
        );
    }
}
