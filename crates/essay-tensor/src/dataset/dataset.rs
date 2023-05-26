use core::fmt;
use std::{collections::VecDeque};

use crate::{Tensor, 
    flow::{
        FlowData, FlowIn, SourceFactory, SourceId, Flow, FlowSingle, 
        FlowSourcesBuilder, FlowBuilderSingle, FlowOutputBuilder, Source, self, Out,
    }, 
};

use super::{take::Take};

pub struct Dataset<T: FlowData = Tensor<f32>> {
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

pub struct DatasetIter<'a, T: FlowData = Tensor<f32>> {
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

impl<T: FlowData> From<FlowSingle<(), T>> for Dataset<T> {
    fn from(flow: FlowSingle<(), T>) -> Self {
        Self {
            flow,
        }
    }
}
