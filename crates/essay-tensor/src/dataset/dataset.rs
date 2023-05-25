use core::fmt;
use std::{marker::PhantomData, collections::VecDeque};

use crate::{Tensor, 
    flow::{
        FlowData, FlowIn, SourceFactory, SourceId, Flow, FlowSingle, 
        FlowSourcesBuilder, FlowBuilderSingle, FlowOutputBuilder, Source, self, Out,
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

//
// DatasetIter
//

pub struct DatasetIter<'a, T: FlowData> {
    iter: <FlowSingle<(), T> as Flow<(), T>>::Iter<'a>,
}

impl<T: FlowData> Iterator for DatasetIter<'_, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
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

//
// constant converters
//

pub fn from_tensors<T: FlowData>(into_dataset: impl Into<Dataset<Tensor<T>>>) -> Dataset<Tensor<T>> {
    into_dataset.into()
}

impl<T: FlowData> From<Tensor<T>> for Dataset<Tensor<T>> {
    fn from(value: Tensor<T>) -> Self {
        let vec = vec![value];
    
        Dataset::from_flow(|builder| {
            builder.source(move || {
                vec.clone()
            }, &())
        })
    }
}

impl<T: FlowData> From<&Tensor<T>> for Dataset<Tensor<T>> {
    fn from(value: &Tensor<T>) -> Self {
        let vec = vec![value.clone()];
    
        Dataset::from_flow(|builder| {
            builder.source(move || {
                vec.clone()
            }, &())
        })
    }
}

impl<T: FlowData> From<Vec<T>> for Dataset<T> {
    fn from(vec: Vec<T>) -> Self {
        let mut vec = vec;
        vec.reverse();
    
        Dataset::from_flow(|builder| {
            builder.source(move || {
                vec.clone()
            }, &())
        })
    }
}

impl<T: FlowData> Source<(), T> for Vec<T> {
    fn next(&mut self, _input: &mut ()) -> flow::Result<Out<T>> {
        Ok(Out::from(self.pop()))
    }
}

impl<T: FlowData> Source<(), T> for VecDeque<T> {
    fn next(&mut self, _input: &mut ()) -> flow::Result<Out<T>> {
        Ok(Out::from(self.pop_front()))
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

        let mut ds = dataset::from_tensors(&t);
        let tensors : Vec<Tensor> = ds.iter().collect();

        assert_eq!(&tensors, &vec![tf32!([1., 2., 3.])]);
    }
}
