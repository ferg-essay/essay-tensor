use crate::{Tensor, tensor::Dtype};

use super::{take::Take};

pub trait Dataset<T:Dtype> : Clone + Sized {
    type IntoIter:Iterator<Item=Tensor<T>>;

    fn iter(&self) -> Self::IntoIter;

    fn get_single_element(&self) -> Tensor<T>;

    fn take(self, count: usize) -> Take<T, Self> {
        Take::new(self, count)
    }
}

impl<T:Dtype> Dataset<T> for Tensor<T> {
    type IntoIter = TensorIter<T>;

    fn iter(&self) -> Self::IntoIter {
        TensorIter {
            tensor: Some(self.clone())
        }
    }

    fn get_single_element(&self) -> Tensor<T> {
        self.clone()
    }
}

pub struct TensorIter<T:Dtype> {
    tensor: Option<Tensor<T>>,
}

impl<T:Dtype> Iterator for TensorIter<T> {
    type Item = Tensor<T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.tensor.take()
    }
}

