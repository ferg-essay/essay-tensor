use crate::{Tensor};

use super::{take::Take};

pub trait Dataset<T:Clone> : Clone + Sized {
    type IntoIter: Iterator<Item=Tensor<T>>;

    fn iter(&self) -> Self::IntoIter;

    fn get_single_element(&self) -> Tensor<T>;

    fn take(self, count: usize) -> Take<T, Self> {
        Take::new(self, count)
    }
}

pub fn from_tensors<T:Clone, I:IntoDataset<T>>(item: I) -> I::Item {
    I::into_dataset(item)
}

#[derive(Clone)]
pub struct DatasetTensorList<T:Clone> {
    tensors: Vec<Tensor<T>>,
}

pub struct TensorIter<T:Clone> {
    tensor: Vec<Tensor<T>>,
}

pub trait IntoDataset<T:Clone> {
    type Item: Dataset<T>;

    fn into_dataset(self) -> Self::Item;
}
/*
impl<T:Clone, D:Dataset<T>> IntoIterator for D {
    type Item = Tensor<T>;
    type IntoIter = D::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
       self.iter()
    }
}
*/
/*
impl<T:Clone> Dataset<T> for Tensor<T> {
    type IntoIter = TensorIter<T>;

    fn iter(&self) -> Self::IntoIter {
        TensorIter {
            tensor: vec![self.clone()]
        }
    }

    fn get_single_element(&self) -> Tensor<T> {
        self.clone()
    }
}
*/

impl<T:Clone> IntoDataset<T> for &Tensor<T> {
    type Item = DatasetTensorList<T>;

    fn into_dataset(self) -> Self::Item {
        DatasetTensorList::from_slice(&[self.clone()])
    }
}

impl<T:Clone> DatasetTensorList<T> {
    pub fn from_slice(vec: &[Tensor<T>]) -> Self {
        let mut dataset_vec = Vec::new();
        let shape = vec[0].shape();

        for i in vec.len() - 1 ..= 0 {
            assert_eq!(&shape, &vec[i].shape());

            dataset_vec.push(vec[i].clone());
        }

        Self {
            tensors: dataset_vec,
        }
    }
}

impl<T:Clone> IntoDataset<T> for &[Tensor<T>] {
    type Item = DatasetTensorList<T>;

    fn into_dataset(self) -> Self::Item {
        DatasetTensorList::from_slice(self)
    }
}

impl<T:Clone> Dataset<T> for DatasetTensorList<T> {
    type IntoIter = TensorIter<T>;

    fn iter(&self) -> Self::IntoIter {
        TensorIter {
            tensor: self.tensors.clone(),
        }
    }

    fn get_single_element(&self) -> Tensor<T> {
        self.tensors[0].clone()
    }
}

impl<T:Clone> Iterator for TensorIter<T> {
    type Item = Tensor<T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.tensor.pop()
    }
}

#[cfg(test)]
mod test {
    use crate::{prelude::*, data};

    #[test]
    fn from_tensors() {
        let t = tf32!([1., 2., 3.]);

        let ds = data::from_tensors(&t);
        let tensors : Vec<Tensor> = ds.iter().collect();

        assert_eq!(&tensors, &vec![tf32!([1., 2., 3.])]);
    }
}
