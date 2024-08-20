mod source;
mod flow_single;
pub mod thread_pool;
mod pipe;
mod flow;
mod data;
mod dispatch;
mod flow_pool;

pub use data::{
    FlowIn, FlowData,
};

pub use pipe::In;

//pub use flow_pool::{
//    FlowPool, PoolFlowBuilder,
//};

pub use flow_single::{
    FlowSingle, FlowBuilderSingle, FlowIterSingle,
};
    
pub use source::{
    SourceId, Source, SourceFactory, Out, Result, VecSource,
};

pub use flow::{
    FlowOutputBuilder, Flow, FlowSourcesBuilder, 
};