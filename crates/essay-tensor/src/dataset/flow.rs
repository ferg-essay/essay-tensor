use crate::flow::{FlowData, SourceId};

use super::{Dataset, IntoFlowBuilder};

pub struct DatasetFlowBuilder {
}

impl DatasetFlowBuilder {
    pub fn new() -> Self {
        Self {

        }
    }

    pub(crate) fn build<T: FlowData>(&self, id: SourceId<T>) -> Dataset<T> {
        todo!()
    }
}
