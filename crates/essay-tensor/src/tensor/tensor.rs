use core::fmt;
use std::{ptr::NonNull, alloc::Layout, alloc, mem, cmp::max, any::type_name, fmt::Display};

pub trait TensorValue : Copy + Display {}

pub struct Tensor<const N:usize,D:TensorValue=f32> {
    shape: [usize; N],
    data: NonNull<D>,
    len: usize,
}

impl<const N:usize,D:TensorValue> Tensor<N,D> {
    pub unsafe fn new_uninit(shape: [usize; N]) -> Self {
        let len: usize = max(1, shape.iter().product());
        let layout = Layout::array::<D>(len).unwrap();
        
        let data =
            NonNull::<D>::new_unchecked(alloc::alloc(layout).cast::<D>());
        
        Self {
            shape,
            data,
            len,
        }
    }

    pub(crate) fn at_offset(&self, offset: usize) -> D {
        assert!(offset < self.len);

        unsafe { *self.data.as_ptr().add(offset) }
    }

    pub(crate) fn set_offset(&self, offset: usize, value: D) {
        assert!(offset < self.len);

        unsafe { *self.data.as_ptr().add(offset) = value; }
    }
}

impl<const N:usize,D:TensorValue> fmt::Debug for Tensor<N, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor{{")?;

        fmt_tensor_rec(&self, f, self.shape.len(), 0)?;

        let mut v : Vec<usize> = Vec::from(self.shape);
        v.reverse();
        
        write!(f, ", shape: {:?}", &v)?;
        write!(f, ", dtype: {}", type_name::<D>())?;

        write!(f, "}}")?;
        Ok(())
    }
}

fn fmt_tensor_rec<const N:usize, D:TensorValue>(
    tensor: &Tensor<N, D>, 
    f: &mut fmt::Formatter<'_>, 
    i: usize,
    offset: usize
) -> fmt::Result {
    match i {
        0 => write!(f, "{}", tensor.at_offset(offset)),
        1 => {
            write!(f, "[")?;

            for j in 0..tensor.shape[0] {
                if j > 0 {
                    write!(f, " ")?;
                }

                fmt_tensor_rec(tensor, f, i - 1, offset + j)?;
            }

            write!(f, "]")
        },
        2 => {
            write!(f, "[")?;

            let stride = tensor.shape[0];
            for j in 0..tensor.shape[1] {
                if j > 0 {
                    write!(f, ",\n ")?;
                }

                fmt_tensor_rec::<N, D>(tensor, f, i - 1, offset + j * stride)?;
            }

            write!(f, "]")
        },
        i => {
            write!(f, "[")?;

            // TODO:
            let stride = tensor.shape[0] * tensor.shape[1];
            for j in 0..tensor.shape[2] {
                if j > 0 {
                    write!(f, ",\n\n  ")?;
                }

                fmt_tensor_rec::<N, D>(tensor, f, i - 1, offset + j * stride)?;
            }

            write!(f, "]")
        },
    }
}

impl<D:TensorValue> From<D> for Tensor<0, D> {
    fn from(value: D) -> Self {
        unsafe {
            let tensor = Tensor::<0, D>::new_uninit([]);
            tensor.set_offset(0, value);
            tensor
        }
    }
}

impl<D:TensorValue, const N:usize> From<[D; N]> for Tensor<1, D> {
    fn from(value: [D; N]) -> Self {
        unsafe {
            let tensor = Tensor::<1, D>::new_uninit([N]);

            for (i, value) in value.iter().enumerate() {
                tensor.set_offset(i, *value);
            }

            tensor
        }
    }
}

impl<D:TensorValue, const N: usize, const M: usize> From<[[D; N]; M]> for Tensor<2, D> {
    fn from(value: [[D; N]; M]) -> Self {
        unsafe {
            let tensor = Tensor::<2, D>::new_uninit([N, M]);

            for (j, value) in value.iter().enumerate() {
                for (i, value) in value.iter().enumerate() {
                    tensor.set_offset(j * N + i, *value);
                }
            }

            tensor
        }
    }
}

impl<D:TensorValue, const N: usize, const M: usize, const L: usize>
    From<[[[D; N]; M]; L]> for Tensor<3, D> {
    fn from(value: [[[D; N]; M]; L]) -> Self {
        unsafe {
            let tensor = Tensor::<3, D>::new_uninit([N, M, L]);

            for (k, value) in value.iter().enumerate() {
                for (j, value) in value.iter().enumerate() {
                    for (i, value) in value.iter().enumerate() {
                      tensor.set_offset(k * N * M + j * N + i, *value);
                    }
                }
            }

            tensor
        }
    }
}

impl TensorValue for f32 {}

#[cfg(test)]
mod test {
    use crate::tensor;

    use super::Tensor;

    #[test]
    fn debug_tensor_from_f32() {
        let t = Tensor::<0>::from(10.5);

        assert_eq!(format!("{:?}", t), "Tensor{10.5, shape: [], dtype: f32}");
    }

    #[test]
    fn debug_vector_from_slice_f32() {
        let t = Tensor::<1>::from([]);
        assert_eq!(format!("{:?}", t), "Tensor{[], shape: [0], dtype: f32}");

        let t = Tensor::<1>::from([10.5]);
        assert_eq!(format!("{:?}", t), "Tensor{[10.5], shape: [1], dtype: f32}");

        let t = Tensor::<1>::from([1., 2.]);
        assert_eq!(format!("{:?}", t), "Tensor{[1 2], shape: [2], dtype: f32}");

        let t = Tensor::<1>::from([1., 2., 3., 4., 5.]);
        assert_eq!(format!("{:?}", t), "Tensor{[1 2 3 4 5], shape: [5], dtype: f32}");
    }

    #[test]
    fn debug_matrix_from_slice_f32() {
        let t = Tensor::<2>::from([[]]);
        assert_eq!(format!("{:?}", t), "Tensor{[[]], shape: [1, 0], dtype: f32}");

        let t = Tensor::<2>::from([[10.5]]);
        assert_eq!(format!("{:?}", t), "Tensor{[[10.5]], shape: [1, 1], dtype: f32}");

        let t = Tensor::<2>::from([[1., 2.]]);
        assert_eq!(format!("{:?}", t), "Tensor{[[1 2]], shape: [1, 2], dtype: f32}");

        let t = Tensor::<2>::from([[1., 2., 3.], [4., 5., 6.]]);
        assert_eq!(format!("{:?}", t), "Tensor{[[1 2 3],\n [4 5 6]], shape: [2, 3], dtype: f32}");
    }

    #[test]
    fn debug_tensor3_from_slice_f32() {
        let t = Tensor::<3>::from([[[]]]);
        assert_eq!(format!("{:?}", t), "Tensor{[[[]]], shape: [1, 1, 0], dtype: f32}");

        let t = Tensor::<3>::from([[[10.5]]]);
        assert_eq!(format!("{:?}", t), "Tensor{[[[10.5]]], shape: [1, 1, 1], dtype: f32}");

        let t = Tensor::<3>::from([[[1., 2.]],[[101., 102.]]]);
        assert_eq!(format!("{:?}", t), "Tensor{[[[1 2]],\n\n  [[101 102]]], shape: [2, 1, 2], dtype: f32}");

        let t = Tensor::<3>::from([[[1., 2.], [3., 4.]],[[101., 102.], [103., 104.]]]);
        assert_eq!(format!("{:?}", t), "Tensor{[[[1 2],\n [3 4]],\n\n  [[101 102],\n [103 104]]], shape: [2, 2, 2], dtype: f32}");

        /*        let t = Tensor::<2>::from([[1., 2., 3.], [4., 5., 6.]]);
        assert_eq!(format!("{:?}", t), "Tensor{[[1 2 3],\n [4 5 6]], shape: [2, 3], dtype: f32}");
        */
    }

    #[test]
    fn debug_vector_from_macro() {
        let t = tensor!(1.);
        assert_eq!(format!("{:?}", t), "Tensor{1, shape: [], dtype: f32}");

        let t = tensor!([1., 2.]);
        assert_eq!(format!("{:?}", t), "Tensor{[1 2], shape: [2], dtype: f32}");

        let t = tensor!([[1., 2., 3.], [3., 4., 5.]]);
        assert_eq!(format!("{:?}", t), "Tensor{[[1 2 3],\n [3 4 5]], shape: [2, 3], dtype: f32}");
/*
        let t = tensor!([
            [[1., 2.], [3., 4.], [5., 6.]],
            [[11., 12.], [13., 14.], [15., 16.]],
        ]);
        assert_eq!(format!("{:?}", t), "Tensor{[[1 2 3],\n [3 4 5]], shape: [2, 3], dtype: f32}");
        */
    }
}