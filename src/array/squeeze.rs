
//
// squeeze operation
//

use crate::{Tensor, prelude::AxisOpt, tensor::{TensorId, Dtype}, model::Operation};

pub fn squeeze<D: Dtype>(x: &Tensor<D>, axis: impl Into<AxisOpt>) -> Tensor<D> {
    let axis : AxisOpt = axis.into();
    let op = SqueezeOp(axis.get_axis());

    //let node = NodeOp::new(&[x], Box::new(op.clone()));
    let id = TensorId::unset();

    let tensor = op.f(&[&x], id);

    //Tape::set_tensor(tensor)
    tensor
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

impl<D: Dtype> Operation<D> for SqueezeOp {
    fn f(
        &self,
        args: &[&Tensor<D>],
        id: TensorId,
    ) -> Tensor<D> {
        let tensor = args[0];

        let shape_slice = tensor.shape().as_slice();
        let mut vec = Vec::<usize>::new();
        match self.axis() {
            None => {
                for dim in shape_slice {
                    if *dim != 1 {
                        vec.push(*dim)
                    }
                }
            },
            Some(axis) => {
                let axis = (axis + shape_slice.len() as isize) % shape_slice.len() as isize;
                let axis = axis as usize;
                
                let mut vec = Vec::<usize>::new();
                for (i, dim) in shape_slice.iter().enumerate() {
                    if i != axis || *dim != 1 {
                        vec.push(*dim)
                    }
                }
            }
        };

        tensor.clone_with_shape(vec, id)
    }
}

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
