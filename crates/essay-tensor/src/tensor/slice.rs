use crate::{Tensor};


pub trait TensorSlice {
    fn slice<T:Clone>(self, tensor: &Tensor<T>) -> Tensor<T>;
}

impl TensorSlice for usize {
    fn slice<T:Clone>(self, tensor: &Tensor<T>) -> Tensor<T> {
        assert!(tensor.rank() > 0);
        assert!(self < tensor.dim(0));

        let shape = tensor.shape().slice(1..);
        let len : usize = shape.len();

        tensor.subslice(self * len, len, shape)
    }
}

impl TensorSlice for (usize, usize) {
    fn slice<T:Clone>(self, tensor: &Tensor<T>) -> Tensor<T> {
        assert!(tensor.rank() > 1);
        assert!(self.0 < tensor.dim(0));
        assert!(self.1 < tensor.dim(1));

        let shape = tensor.shape().slice(2..);
        let len : usize = shape.len();
        let offset = (self.0 * tensor.dim(1) + self.1) * len;

        tensor.subslice(offset, len, shape)
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    use super::{Tensor};

    #[test]
    fn slice_usize() {
        let t = tf32!([[1., 2.], [3., 4.], [5., 6.]]);
        assert_eq!(t.shape().as_slice(), &[3, 2]);

        let t1 = t.slice(0);
        assert_eq!(t1.shape().as_slice(), &[2]);
        assert_eq!(t1.as_slice(), &[1., 2.]);

        let t1 = t.slice(1);
        assert_eq!(t1.shape().as_slice(), &[2]);
        assert_eq!(t1.as_slice(), &[3., 4.]);

        let t1 = t.slice(2);
        assert_eq!(t1.shape().as_slice(), &[2]);
        assert_eq!(t1.as_slice(), &[5., 6.]);
    }

    #[test]
    fn slice_usize_usize() {
        let t = tf32!([
            [[1., 2.], [3., 4.], [5., 6.]],
            [[10., 20.], [30., 40.], [50., 60.]],
        ]);
        assert_eq!(t.shape().as_slice(), &[2, 3, 2]);

        let t1 = t.slice((0, 1));
        assert_eq!(t1.len(), 2);
        assert_eq!(t1.as_slice(), &[3., 4.]);

        let t1 = t.slice((1, 2));
        assert_eq!(t1.len(), 2);
        assert_eq!(t1.as_slice(), &[50., 60.]);
    }
}