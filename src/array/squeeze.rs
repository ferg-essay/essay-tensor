
//
// squeeze operation
//

use crate::{Tensor, prelude::AxisOpt, tensor::{TensorId, Dtype}};

pub fn squeeze<D: Dtype>(x: &Tensor<D>, axis: impl Into<AxisOpt>) -> Tensor<D> {
    let axis : AxisOpt = axis.into();
    let op = SqueezeOp(axis.get_axis());

    //let node = NodeOp::new(&[x], Box::new(op.clone()));
    let id = TensorId::unset();

    //let tensor = op.f(&[&x], id);

    //Tape::set_tensor(tensor)
    //    tensor
    todo!();
}

impl<D: Dtype> Tensor<D> {
    pub fn squeeze(x: &Tensor<D>, axis: impl Into<AxisOpt>) -> Tensor<D> {
        squeeze(x, axis)
    }
}

#[derive(Clone)]
pub struct SqueezeOp(Option<isize>);

impl SqueezeOp {
    fn axis(&self) -> &Option<isize> {
        &self.0
    }
}
/*
impl<D: Dtype> Operation<D> for SqueezeOp {
    fn f(
        &self,
        args: &[&Tensor<D>],
        id: TensorId,
    ) -> Tensor<D> {
        let tensor = args[0];

        let shape = tensor.shape().squeeze(self.axis());

        tensor.clone_with_shape(shape, id)
    }
}
*/

#[cfg(test)]
mod test {
    use crate::{prelude::*, array::{squeeze, Axis}};
    
    #[test]
    fn test_squeeze() {
        assert_eq!(squeeze(&tf32!([[1.]]), ()), tf32!(1.));
        assert_eq!(squeeze(&tf32!([[1., 2.]]), ()), tf32!([1., 2.]));
        assert_eq!(squeeze(&tf32!([[[1.], [2.]]]), ()), tf32!([1., 2.]));
    }
    
    #[test]
    fn test_squeeze_axis() {
        assert_eq!(squeeze(&tf32!([[[1.], [2.]]]), Axis::axis(-1)), tf32!([[1., 2.]]));
    }
}
