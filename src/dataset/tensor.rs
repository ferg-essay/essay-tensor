
//
// constant converters
//

use core::fmt;

use crate::{Tensor, flow::{FlowData, Source, VecSource}, tensor::Dtype};

use super::Dataset;

pub fn from_tensors<T: Dtype>(into_dataset: impl Into<Dataset<Tensor<T>>>) -> Dataset<Tensor<T>> {
    into_dataset.into()
}

pub fn from_tensor_slices<T: Dtype>(into_tensor: impl Into<Tensor<T>>) -> Dataset<Tensor<T>> {
    let mut vec: Vec<Tensor<T>> = Vec::new();

    let tensor = into_tensor.into();

    for i in 0..tensor.dim(0) {
        vec.push(tensor.slice(i));
    }

    Dataset::from(vec)
}

impl<T: Dtype> From<Tensor<T>> for Dataset<Tensor<T>> {
    fn from(value: Tensor<T>) -> Self {
        let vec = vec![value];
    
        Dataset::from_flow(|builder| {
            builder.source(move || {
                // TODO: cleanup so vec.clone() is sufficient
                VecSource::from(vec.clone())
            }, &())
        })
    }
}

impl<T: Dtype> From<&Tensor<T>> for Dataset<Tensor<T>> {
    fn from(value: &Tensor<T>) -> Self {
        let vec = vec![value.clone()];
    
        Dataset::from_flow(|builder| {
            builder.source(move || {
                VecSource::from(vec.clone())
            }, &())
        })
    }
}

impl<T: Dtype> From<Vec<Tensor<T>>> for Dataset<Tensor<T>> {
    fn from(vec: Vec<Tensor<T>>) -> Self {
        let mut vec = vec;
        vec.reverse();
    
        Dataset::from_flow(|builder| {
            builder.source::<(), Tensor<T>>(move || {
                VecSource::from(vec.clone())
            }, &())
        })
    }
}

impl<T: Send + Sync + fmt::Debug + 'static> FlowData for Tensor<T> {}

#[cfg(test)]
mod test {
    use crate::{prelude::*, dataset};

    #[test]
    fn test_from_tensors() {
        let t = tf32!([1., 2., 3.]);

        let mut ds = dataset::from_tensors(&t);
        let tensors : Vec<Tensor> = ds.iter().collect();

        assert_eq!(&tensors, &vec![tf32!([1., 2., 3.])]);
    }

    #[test]
    fn test_from_tensor_slices_rank1() {
        let t = tf32!([1., 2., 3., 4.]);

        let mut ds = dataset::from_tensor_slices(&t);
        let tensors : Vec<Tensor> = ds.iter().collect();

        assert_eq!(&tensors, &vec![tf32!(1.), tf32!(2.), tf32!(3.), tf32!(4.)]);
    }

    #[test]
    fn test_from_tensor_slices_rank2() {
        let t = tf32!([[1., 2.], [3., 4.]]);

        let mut ds = dataset::from_tensor_slices(&t);
        let tensors : Vec<Tensor> = ds.iter().collect();

        assert_eq!(&tensors, &vec![tf32!([1., 2.]), tf32!([3., 4.])]);
    }
}

