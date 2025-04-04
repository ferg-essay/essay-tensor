use crate::tensor::{Dtype, Tensor};

pub fn flatten<D: Dtype>(tensor: impl AsRef<Tensor<D>>) -> Tensor<D> {
    let tensor = tensor.as_ref();
    tensor.clone().reshape([tensor.shape().size()])
}

impl<D: Dtype> Tensor<D> {
    #[inline]
    pub fn flatten(&self) -> Tensor<D> {
        flatten(self)
    }
}

#[cfg(test)]
mod test {
    use crate::{prelude::*, array::flatten};
    
    #[test]
    fn test_flatten() {
        assert_eq!(flatten(&tf32!([[1., 2.], [3., 4.]])), tf32!([1., 2., 3., 4.]));
    }
}
