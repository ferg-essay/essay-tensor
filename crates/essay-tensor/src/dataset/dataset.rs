use std::marker::PhantomData;

use crate::{Tensor, flow::{FlowData, FlowIn, SourceFactory, SourceId}};

use super::{take::Take};

pub struct Dataset<T: FlowData> {
    marker: PhantomData<T>,
}

impl<T: FlowData> Dataset<T> {
    pub fn iter(&self) -> DatasetIter<T> {
        todo!()
    }

    pub fn get_single_element(&self) -> T {
        todo!()
    }

    pub fn take(self, count: usize) -> Dataset<T> {
        Self::from_flow(|builder| {
            let input = self.into_flow(builder);

            Take::build(builder, input, count)
        })
    }

    pub fn from_flow(fun: impl FnOnce(&mut IntoFlowBuilder) -> SourceId<T>) -> Self {
        let mut builder = IntoFlowBuilder::new();

        let id = fun(&mut builder);

        builder.build_dataset(id)
    }
}

impl<T: FlowData> IntoFlow<T> for Dataset<T> {
    fn into_flow(self, builder: &mut IntoFlowBuilder) -> SourceId<T> {
        todo!()
    }
}

pub struct DatasetIter<T: FlowData> {
    marker: PhantomData<T>,
}

impl<T: FlowData> Iterator for DatasetIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

/*
pub trait Dataset<T: FlowData> : Clone + Send + Sync + Sized + 'static {
    type IntoIter: Iterator<Item=Tensor<T>>;

    fn iter(&self) -> Self::IntoIter;

    fn get_single_element(&self) -> Tensor<T>;
    
    //fn take(self, count: usize) -> Take<T> {
    //    todo!()
    //    // Take::new(self, count)
    //}
}
*/

pub fn from_tensors<T: FlowData, I: IntoDataset<Tensor<T>>>(item: I) -> Dataset<Tensor<T>> {
    I::into_dataset(item)
}

pub struct TensorIter<T:Clone> {
    tensor: Vec<Tensor<T>>,
}

pub trait IntoDataset<T: FlowData> {
    // type Item: Dataset<T>;

    fn into_dataset(self) -> Dataset<T>;
}

impl<T: FlowData> IntoDataset<T> for &Tensor<T> {
    // type Item = DatasetTensorList<T>;

    fn into_dataset(self) -> Dataset<T> {
        todo!()
        // DatasetTensorList::from_slice(&[self.clone()])
    }
}

#[derive(Clone)]
pub struct DatasetTensorList<T:Clone> {
    tensors: Vec<Tensor<T>>,
}

impl<T: FlowData> DatasetTensorList<T> {
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

impl<T: FlowData> IntoDataset<Tensor<T>> for &[Tensor<T>] {
    // type Item = DatasetTensorList<T>;

    fn into_dataset(self) -> Dataset<Tensor<T>> {
        // DatasetTensorList::from_slice(self)
        todo!()
    }
}
/*
impl<T: FlowData> Dataset<T> for DatasetTensorList<T> {
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
*/

impl<T:Clone> Iterator for TensorIter<T> {
    type Item = Tensor<T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.tensor.pop()
    }
}

pub trait IntoFlow<T: FlowData> {
    fn into_flow(self, builder: &mut IntoFlowBuilder) -> SourceId<T>;
}

pub struct IntoFlowBuilder {

}

impl IntoFlowBuilder {
    pub fn new() -> Self {
        Self {

        }
    }

    pub fn source<I, O>(
        &mut self,
        source: impl SourceFactory<I, O>,
        in_nodes: &I::Nodes,
    ) -> SourceId<O>
    where
        I: FlowIn<I>,
        O: FlowData,
    {
        todo!()
    }

    fn build_dataset<T: FlowData>(&self, id: SourceId<T>) -> Dataset<T> {
        todo!()
    }
}

impl<T: Send + Sync + 'static> FlowData for Tensor<T> {}

#[cfg(test)]
mod test {
    use crate::{prelude::*, dataset};

    #[test]
    fn from_tensors() {
        let t = tf32!([1., 2., 3.]);
        todo!();
        /*
        let ds = dataset::from_tensors(&t);
        let tensors : Vec<Tensor> = ds.iter().collect();

        assert_eq!(&tensors, &vec![tf32!([1., 2., 3.])]);
        */
    }
}
