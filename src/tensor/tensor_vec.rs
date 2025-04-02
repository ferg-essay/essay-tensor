
use crate::Tensor;

use super::{Dtype, TensorUninit};

pub struct TensorVec<T> {
    vec: Vec<T>,
}

impl<T> TensorVec<T> {
    #[inline]
    pub fn len(&self) -> usize {
        self.vec.len()
    }

    #[inline]
    pub fn push(&mut self, value: T) {
        self.vec.push(value);
    }

    #[inline]
    pub fn reverse(&mut self) {
        self.vec.reverse();
    }

    #[inline]
    pub fn pop(&mut self) {
        self.vec.pop();
    }

    #[inline]
    pub fn append(&mut self, tail: &mut TensorVec<T>) {
        self.vec.append(&mut tail.vec);
    }
}

impl<T: Dtype> TensorVec<T> {
    #[inline]
    pub fn new() -> Self {
        Self {
            vec: Vec::new(),
        }
    }

    pub fn into_tensor(self) -> Tensor<T> {
        Tensor::from(self.vec)
    }
}

impl<const M: usize, T> TensorVec<[T; M]>
where
    T: Dtype + Copy
{
    #[inline]
    pub fn new() -> Self {
        assert!(M > 0);

        Self {
            vec: Vec::new(),
        }
    }

    pub fn into_tensor(self) -> Tensor<T> {
        unsafe {
            let len = self.vec.len();

            let mut out = TensorUninit::<T>::new(self.vec.len() * M);

            let o = out.as_mut_slice();
            for (j, line) in self.vec.iter().enumerate() {
                for i in 0..M {
                    o[j * M + i] = line[i];
                }
            }

            out.into_tensor([len, M])
        }
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    use super::TensorVec;

    #[test]
    fn test_f32_push() {
        let mut vec = TensorVec::<f32>::new();
        vec.push(1.);
        vec.push(2.);
        assert_eq!(vec.into_tensor(), tf32!([1., 2.]));
    }

    #[test]
    fn test_f32x2_push() {
        let mut vec = TensorVec::<[f32; 2]>::new();
        vec.push([1., 10.]);
        vec.push([2., 20.]);
        assert_eq!(vec.into_tensor(), tf32!([[1., 10.], [2., 20.]]));
    }

    #[test]
    fn vec_f32_push() {
        let mut vec = Vec::<f32>::new();
        vec.push(1.);
        vec.push(2.);
        assert_eq!(Tensor::from(vec), tf32!([1., 2.]));
    }

    #[test]
    fn vec_f32x2_push() {
        let mut vec = Vec::<[f32; 2]>::new();
        vec.push([1., 10.]);
        vec.push([2., 20.]);
        assert_eq!(Tensor::from(vec), tf32!([[1., 10.], [2., 20.]]));
    }

    #[test]
    fn test_vec_drop() {
        todo!();
    }
}