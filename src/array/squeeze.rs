
//
// squeeze operation
//

use crate::tensor::{Axis, Type, Tensor};

pub fn squeeze<D: Type>(tensor: &Tensor<D>) -> Tensor<D> {
    let shape = tensor.shape().squeeze(None);

    tensor.clone().reshape(shape)
}

pub fn squeeze_axis<D: Type>(axis: impl Into<Axis>, tensor: &Tensor<D>) -> Tensor<D> {
    let shape = tensor.shape().squeeze(axis);

    tensor.clone().reshape(shape)
}

impl<D: Type> Tensor<D> {
    pub fn squeeze(&self, axis: impl Into<Axis>) -> Tensor<D> {
        let shape = self.shape().squeeze(axis);

        self.clone().reshape(shape)
    }
}

#[cfg(test)]
mod test {
    use crate::{array::squeeze::{squeeze, squeeze_axis}, tensor::Axis, prelude::*};
    
    #[test]
    fn test_squeeze() {
        assert_eq!(squeeze(&tf32!([[1.]])), tf32!(1.));
        assert_eq!(squeeze(&tf32!([[1., 2.]])), tf32!([1., 2.]));
        assert_eq!(squeeze(&tf32!([[[1.], [2.]]])), tf32!([1., 2.]));
    }
    
    #[test]
    fn test_squeeze_axis() {
        assert_eq!(squeeze_axis(Axis::axis(-1), &tf32!([[[1.], [2.]]])), tf32!([[1., 2.]]));
    }
}
