
//
// expand dims
//

use crate::{Tensor, prelude::AxisOpt, tensor::{TensorId, Dtype}, model::Operation};

pub fn expand_dims<D: Dtype>(x: impl Into<Tensor<D>>, axis: impl Into<AxisOpt>) -> Tensor<D> {
    x.into().expand_dims(axis)
}

impl<D: Dtype> Tensor<D> {
    pub fn expand_dims(&self, axis: impl Into<AxisOpt>) -> Tensor<D> {
        let axis : AxisOpt = axis.into();
        let op = ExpandDims(axis.get_axis().unwrap());
    
        //let node = NodeOp::new(&[x], Box::new(op.clone()));
        let id = TensorId::unset();
    
        let tensor = op.f(&[self], id);
    
        D::set_tape(tensor)
    }
}

#[derive(Clone)]
pub struct ExpandDims(isize);

impl ExpandDims {
    #[inline]
    fn axis(&self) -> isize {
        self.0
    }
}

impl<D: Dtype> Operation<D> for ExpandDims {
    fn f(
        &self,
        args: &[&Tensor<D>],
        id: TensorId,
    ) -> Tensor<D> {
        let tensor = args[0];

        let mut vec = Vec::from(tensor.shape().as_slice());
        let axis = self.axis();

        let index = if axis >= 0  { axis } else { vec.len() as isize + axis + 1 } as usize;
        assert!(index <= vec.len(), "expand_dims axis is invalid {} in {:?}", axis, tensor.shape().as_slice());

        vec.insert(index, 1);

        tensor.clone_with_shape(vec, id)
    }
}

#[cfg(test)]
mod test {
    use crate::{prelude::*, array::{squeeze, Axis, expand_dims}};
    
    #[test]
    fn test_expand_dims() {
        assert_eq!(expand_dims(&tf32!([1., 2.]), 0), tf32!([[1., 2.]]));
        assert_eq!(expand_dims(&tf32!([1., 2.]), 1), tf32!([[1.], [2.]]));
        assert_eq!(expand_dims(&tf32!([1., 2.]), -1), tf32!([[1.], [2.]]));

        assert_eq!(
            expand_dims(&tf32!([[1., 2.], [3., 4.]]), 0), 
            tf32!([[[1., 2.], [3., 4.]]])
        );

        assert_eq!(
            expand_dims(&tf32!([[1., 2.], [3., 4.]]), 1), 
            tf32!([[[1., 2.]], [[3., 4.]]])
        );

        assert_eq!(
            expand_dims(&tf32!([[1., 2.], [3., 4.]]), 2), 
            tf32!([[[1.], [2.]], [[3.], [4.]]])
        );

        assert_eq!(
            expand_dims(&tf32!([[1., 2.], [3., 4.]]), -1), 
            tf32!([[[1.], [2.]], [[3.], [4.]]])
        );

        assert_eq!(
            expand_dims(&tf32!([[1., 2.], [3., 4.]]), -2), 
            tf32!([[[1., 2.]], [[3., 4.]]])
        );
    }
}
