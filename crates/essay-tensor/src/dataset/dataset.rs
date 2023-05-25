use core::fmt;
use std::marker::PhantomData;

use crate::{Tensor, 
    flow::{
        FlowData, FlowIn, SourceFactory, SourceId, Flow, FlowSingle, 
        FlowSourcesBuilder, FlowBuilderSingle, FlowOutputBuilder,
    }
};

use super::{take::Take};

pub struct Dataset<T: FlowData> {
    // marker: PhantomData<T>,

    flow: FlowSingle<(), T>,
}

impl<T: FlowData> Dataset<T> {
    pub fn iter(&mut self) -> DatasetIter<T> {
        DatasetIter {
            iter: self.flow.iter(())
        }
    }

    pub fn get_single_element(&mut self) -> T {
        self.flow.call(()).unwrap()
    }

    pub fn take(self, count: usize) -> Dataset<T> {
        Self::from_flow(|builder| {
            //let input = self.into_flow(builder);
            let id = builder.add_flow(self.flow);

            Take::build(builder, id, count)
        })
    }

    pub fn from_flow(fun: impl FnOnce(&mut IntoFlowBuilder) -> SourceId<T>) -> Self {
        let mut builder = IntoFlowBuilder::new();

        let id = fun(&mut builder);

        builder.build_dataset(id)
    }
}
/*
impl<T: FlowData> IntoFlow<T> for Dataset<T> {
    fn into_flow(self, builder: &mut IntoFlowBuilder) -> SourceId<T> {
        //self.flow.export(builder)
        todo!()
    }
}
*/

pub struct DatasetIter<'a, T: FlowData> {
    iter: <FlowSingle<(), T> as Flow<(), T>>::Iter<'a>,
}

impl<T: FlowData> Iterator for DatasetIter<'_, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

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

//
// IntoFlow
//

pub trait IntoFlow<T: FlowData> {
    fn into_flow(self, builder: &mut IntoFlowBuilder) -> SourceId<T>;
}

pub struct IntoFlowBuilder {
    builder: FlowBuilderSingle<()>,
}

impl IntoFlowBuilder {
    pub fn new() -> Self {
        Self {
            builder: FlowBuilderSingle::<()>::new()
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
        self.builder.source(source, in_nodes)
    }

    fn add_flow<O>(
        &mut self,
        flow: FlowSingle<(), O>,
    ) -> SourceId<O>
    where
        O: FlowData
    {
        flow.export(&mut self.builder.sources())
    }

    fn build_dataset<T: FlowData>(self, id: SourceId<T>) -> Dataset<T> {
        let flow = self.builder.output(&id);

        Dataset::from(flow)
    }
}

impl<T: FlowData> From<FlowSingle<(), T>> for Dataset<T> {
    fn from(flow: FlowSingle<(), T>) -> Self {
        Self {
            flow,
        }
    }
}

impl<T: Send + Sync + fmt::Debug + 'static> FlowData for Tensor<T> {}

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
