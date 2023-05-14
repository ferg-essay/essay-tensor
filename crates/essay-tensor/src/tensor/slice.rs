use crate::{Tensor};


pub trait TensorSlice {
    fn slice<T:Clone>(self, tensor: &Tensor<T>) -> Tensor<T>;
}

impl TensorSlice for usize {
    fn slice<T:Clone>(self, tensor: &Tensor<T>) -> Tensor<T> {
        assert!(tensor.rank() > 0);
        assert!(self < tensor.dim(0));

        let len : usize = tensor.shape()[1..].iter().product();
        /* 
        let slice = tensor.as_wrap_slice(self * size .. (self + 1) * size);

        let data = TensorData::<T>::from(slice);
        */

        tensor.subslice(self * len, len, &tensor.shape()[1..])

        //Tensor::<T>::new(data, &tensor.shape()[1..])
    }
}

impl TensorSlice for (usize, usize) {
    fn slice<T:Clone>(self, tensor: &Tensor<T>) -> Tensor<T> {
        assert!(tensor.rank() > 1);
        assert!(self.0 < tensor.dim(0));
        assert!(self.1 < tensor.dim(1));

        let size : usize = tensor.shape()[2..].iter().product();
        let offset = (self.0 * tensor.dim(1) + self.1) * size;

        /*
        let slice = tensor.data().as_wrap_slice(offset .. offset + size);

        let data = TensorData::<T>::from(slice);

        Tensor::<T>::new(data, &tensor.shape()[2..])
        */
        tensor.subslice(offset, size, &tensor.shape()[2..])
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    use super::{Tensor};

    #[test]
    fn slice_usize() {
        let t = tf32!([[1., 2.], [3., 4.], [5., 6.]]);
        assert_eq!(t.shape(), &[3, 2]);

        let t1 = t.slice(0);
        assert_eq!(t1.shape(), &[2]);
        assert_eq!(t1.as_slice(), &[1., 2.]);

        let t1 = t.slice(1);
        assert_eq!(t1.shape(), &[2]);
        assert_eq!(t1.as_slice(), &[3., 4.]);

        let t1 = t.slice(2);
        assert_eq!(t1.shape(), &[2]);
        assert_eq!(t1.as_slice(), &[5., 6.]);
    }

    #[test]
    fn slice_usize_usize() {
        let t = tf32!([
            [[1., 2.], [3., 4.], [5., 6.]],
            [[10., 20.], [30., 40.], [50., 60.]],
        ]);
        assert_eq!(t.shape(), &[2, 3, 2]);

        let t1 = t.slice((0, 1));
        assert_eq!(t1.len(), 2);
        assert_eq!(t1.as_slice(), &[3., 4.]);

        let t1 = t.slice((1, 2));
        assert_eq!(t1.len(), 2);
        assert_eq!(t1.as_slice(), &[50., 60.]);
    }
}