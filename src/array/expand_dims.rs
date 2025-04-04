use crate::tensor::{Axis, Dtype, Tensor};

pub fn expand_dims<D: Dtype>(x: impl Into<Tensor<D>>, axis: impl Into<Axis>) -> Tensor<D> {
    x.into().expand_dims(axis)
}

impl<D: Dtype> Tensor<D> {
    pub fn expand_dims(&self, axis: impl Into<Axis>) -> Tensor<D> {
        let axis : Axis = axis.into();

        let axis = axis.get_axis().unwrap_or(0);

        let shape = self.shape().expand_dims(axis);

        self.clone().reshape(shape)
    }
}

#[cfg(test)]
mod test {
    use crate::{prelude::*, array::expand_dims};
    
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
