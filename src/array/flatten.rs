use crate::{
    Tensor, 
    tensor::Dtype, 
};

pub fn flatten<D: Dtype>(x: &Tensor<D>) -> Tensor<D> {
    let op = FlattenOp;

    //let node = NodeOp::new(&[x], Box::new(op.clone()));
    //let id = TensorId::unset();

    //let tensor = op.f(&[&x], id);

    //Tape::set_tensor(tensor)
    //tensor

    todo!();
}

impl<D: Dtype> Tensor<D> {
    #[inline]
    pub fn flatten(&self) -> Tensor<D> {
        flatten(self)
    }
}

#[derive(Clone)]
pub struct FlattenOp;

/*
impl<D: Dtype> Operation<D> for FlattenOp {
    fn f(
        &self,
        args: &[&Tensor<D>],
        id: TensorId,
    ) -> Tensor<D> {
        let tensor = args[0];

        tensor.clone_with_shape([tensor.shape().size()], id)
    }
}
    */

#[cfg(test)]
mod test {
    use crate::{prelude::*, array::{flatten}};
    
    #[test]
    fn test_flatten() {
        assert_eq!(flatten(&tf32!([[1., 2.], [3., 4.]])), tf32!([1., 2., 3., 4.]));
    }
}
