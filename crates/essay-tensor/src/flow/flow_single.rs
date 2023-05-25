use std::{
    any::Any,
    mem, 
    sync::{Arc, Mutex}, marker::PhantomData, 
};

use super::{
    FlowData, FlowIn, Source, 
    Out, pipe::{PipeIn, EagerPipeTrait, EagerPipe}, dispatch::InnerWaker, SourceId, SourceFactory, FlowOutputBuilder, flow::{Flow, FlowSourcesBuilder}, source::{NodeId, SharedOutput, OutputSource, TailSource, UnsetSource}, data::FlowInBuilder,
};


#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PipeId(usize);

// 
// FlowSingle
//

pub struct FlowSingle<I: FlowIn<I>, O: FlowIn<O>> {
    sources: Vec<bool>,
    // sources_inner: SourcesInner,

    _input_ids: I::Nodes,

    _output_id: NodeId,

    output: SharedOutput<O>,
}

impl<I, O> FlowSingle<I, O>
where
    I: FlowIn<I>,
    O: FlowIn<O>
{
    pub fn next(&mut self) -> Option<O> {
        /*
       //  self.output.fill(&mut waker.inner());

        while waker.outer().apply(&mut self.sources_outer) ||
            waker.inner().apply(&mut self.sources_inner) {
        }

        // assert!(self.output.fill(&mut waker.inner()));

        match self.output.take() {
            Some(value) => Some(value),
            None => None,
        }
        */
        todo!()
    }

    fn init(&mut self) {
        //self.output.init();
        self.output.take();
        // self.sources.init();
    }
}
/*
impl<I, O> Flow<I, O> for FlowSingle<I, O>
where
    I: FlowIn<I>,
    O: FlowIn<O>
{
    type Iter<'a> = FlowIterBase<'a, O>;

    fn iter<'a>(&'a mut self, _input: I) -> FlowIterBase<'a, O> {
        // let mut data = self.new_data();

        // In::write(&self.input, &mut data, input);

        self.init(); // &mut data);

        let v = SingleFlowIter {
            flow: self,
            // data: data,
            _count: 0,
        };

        todo!()
    }
}
*/
/*
pub struct FlowIter<'a, I: FlowIn<I>, O: FlowIn<O>> {
    flow: &'a mut FlowSingle<I, O>,
    waker: Dispatcher,
    _count: usize,
}

impl<'a, I: FlowIn<I>, O: FlowIn<O>> FlowIter<'a, I, O> {
    fn new(flow: &'a mut FlowSingle<I, O>) -> Self {
        Self {
            flow,
            waker: Dispatcher::new(),
            _count: 0,
        }
    }
}

impl<I: FlowIn<I>, O: FlowIn<O>> Iterator for FlowIter<'_, I, O> {
    type Item = O;

    fn next(&mut self) -> Option<Self::Item> {
        // self.flow.next()
        todo!()
    }
}
*/

pub struct SingleFlowIter<'a, I: FlowIn<I>, O: FlowIn<O>> {
    flow: &'a mut FlowSingle<I, O>,
    _count: usize,
}

impl<'a, I: FlowIn<I>, O: FlowIn<O>> SingleFlowIter<'a, I, O> {
    fn new(flow: &'a mut FlowSingle<I, O>) -> Self {
        Self {
            flow,
            _count: 0,
        }
    }
}

impl<I: FlowIn<I>, O: FlowIn<O>> Iterator for SingleFlowIter<'_, I, O> {
    type Item = O;

    fn next(&mut self) -> Option<Self::Item> {
        todo!();
    }
}

pub struct EagerFlow<I: FlowIn<I>, O: FlowIn<O>> {
    sources: Vec<Box<dyn EagerSourceTrait>>,
    // sources: EagerSources,

    input: I::Nodes,
    output: SharedOutput<O>,
    tail_id: SourceId<bool>,
}

impl<I: FlowIn<I>, O: FlowIn<O>> Flow<I, O> for EagerFlow<I, O> {
    type Iter<'a> = EagerIter<'a, I, O>;

    fn iter<'a>(&'a mut self, input: I) -> Self::Iter<'a> {
        todo!()
    }
}

pub struct EagerIter<'a, I: FlowIn<I>, O: FlowIn<O>> {
    flow: &'a mut EagerFlow<I, O>,
    marker: PhantomData<O>,
}

impl<I: FlowIn<I>, O: FlowIn<O>> Iterator for EagerIter<'_, I, O> {
    type Item = O;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

pub struct EagerSources {
    sources: Vec<Box<dyn EagerSourceTrait>>,
}

impl EagerSources {
    fn data_request(&self, src_id: NodeId, src_index: usize, n_request: u64) {
        //self.sources[src_id.index()].data_request(src_index, n_request, &self);
        todo!()
    }
}

pub trait EagerSourceTrait {
    fn data_request(&mut self, out_index: usize, n_request: u64, flow: &EagerSources);
}

pub struct EagerSource<I: FlowIn<I>, O: FlowData> {
    factory: Box<dyn SourceFactory<I, O>>,
    source: Box<dyn Source<I, O>>,
    inputs: I::Input,
    input_pipes: Vec<Box<dyn EagerPipeTrait>>,
    output_pipes: Vec<EagerPipe<O>>,

    is_done: bool,
}

impl<I, O> EagerSource<I, O>
where
    I: FlowIn<I>,
    O: FlowData
{
    fn input_request(&mut self, flow: &EagerSources) {
        for pipe in &self.input_pipes {
            if pipe.is_pending() {
                flow.data_request(pipe.src_id(), pipe.src_index(), pipe.n_read() + 1);
            }
        }
    }
}

impl<I, O> EagerSourceTrait for EagerSource<I, O>
where
    I: FlowIn<I>,
    O: FlowData
{
    fn data_request(&mut self, out_index: usize, n_request: u64, flow: &EagerSources) {
        let is_avail = self.output_pipes[out_index].data_request(n_request);

        if is_avail {
            loop {
                self.input_request(flow);

                    // I:fill(inputs)
                match self.source.next(&mut self.inputs) {
                    Ok(Out::Some(value)) => {
                        self.output_pipes[out_index].send(Out::Some(value));
                        return;
                    }
                    Ok(Out::None) => {
                        self.output_pipes[out_index].send(Out::None);
                        return;
                    }
                    Ok(Out::Pending) => {
                        continue;
                    }
                    Err(err) => { panic!("Unknown error {:?}", &err); }
                }
            }
        }
    }
}


//
// Eager builder
//

pub struct EagerFlowBuilder<I: FlowIn<I>> {
    builder: EagerBuilder,
    inputs: Option<I::Nodes>,
    sources: Vec<Box<dyn EagerSourceBuilderTrait>>,
    // marker: PhantomData<O>,
}

impl<In: FlowIn<In>> EagerFlowBuilder<In> {
    pub(crate) fn new() -> Self {
        Self {
            builder: EagerBuilder::new(),
            inputs: None,
            sources: Vec::new(),
            // marker: PhantomData,
        }
    }

    /*
    pub(crate) fn add_pipe<O: 'static>(
        &mut self,
        src_id: SourceId<O>,
        dst_id: NodeId,
        dst_index: usize,
    ) -> Box<dyn PipeIn<O>> {
        let source = &mut self.sources[src_id.index()];

        unsafe { 
            mem::transmute(source.add_pipe(dst_id, dst_index))
        }
    }
    */
}

impl<In: FlowIn<In>> FlowSourcesBuilder for EagerFlowBuilder<In> {
    fn source<I, O>(
        &mut self, 
        into_source: impl SourceFactory<I, O>,
        in_nodes: &I::Nodes,
    ) -> SourceId<O>
    where
        I: FlowIn<I>,
        O: FlowData,
    {
        let id = SourceId::<O>::new(self.sources.len());
        let source = EagerSourceBuilder::new(
            id.clone(), 
            Box::new(into_source),
            in_nodes, 
            self,
        );
    
        assert_eq!(id.index(), self.sources.len());

        self.sources.push(Box::new(source));

        id
    }
}

impl<In: FlowIn<In>> FlowOutputBuilder<In> for EagerFlowBuilder<In> {
    type Flow<O: FlowIn<O>> = EagerFlow<In, O>;
    
    fn output<O: FlowIn<O>>(mut self, src_nodes: &O::Nodes) -> Self::Flow<O> {
        let mut flow_sources = Vec::new();

        for mut source in self.sources.drain(..) {
            source.build(&mut flow_sources)
        }

        let mut output_ids = Vec::new();

        O::node_ids(src_nodes, &mut output_ids);

        let shared_output = SharedOutput::<O>::new();

        let output_ptr = shared_output.clone();

        let id = self.source::<O, bool>(move || {
            OutputSource::new(output_ptr.clone())
        }, src_nodes);
        let tail = self.source::<bool, bool>(|| TailSource, &id);

        EagerFlow {
            sources: flow_sources,

            input: self.inputs.unwrap(),
            output: shared_output, // In::new(source),
            tail_id: id,
        }
    }

    fn input(&mut self) -> &In::Nodes {
        todo!()
    }
}

impl<In: FlowIn<In>> FlowInBuilder for EagerFlowBuilder<In> {
    fn add_source<I, O>(
        &mut self, 
        source: impl SourceFactory<I, O>,
        in_nodes: &I::Nodes,
    ) -> SourceId<O>
    where
        I: FlowIn<I>,
        O: FlowData {
        self.source(source, in_nodes)
    }

    fn add_pipe<O: FlowIn<O>>(
        &mut self,
        src_id: SourceId<O>,
        dst_id: NodeId,
        dst_index: usize,
    ) -> Box<dyn PipeIn<O>> {
        let source = &mut self.sources[src_id.index()];

        unsafe { 
            mem::transmute(source.add_pipe(dst_id, dst_index))
        }
    }
}


pub struct EagerBuilder {
    sources: Vec<Box<dyn EagerSourceBuilderTrait>>,
}

impl EagerBuilder {
    pub(crate) fn new() -> Self {
        Self {
            sources: Vec::default(),
        }
    }

    pub(crate) fn add_pipe<O: 'static>(
        &mut self,
        src_id: SourceId<O>,
        dst_id: NodeId,
        dst_index: usize,
    ) -> Box<dyn PipeIn<O>> {
        let source = &mut self.sources[src_id.index()];

        unsafe { 
            mem::transmute(source.add_pipe(dst_id, dst_index))
        }
    }
}

//
// Eager builder
//

pub(crate) trait EagerSourceBuilderTrait {
    unsafe fn add_pipe(&mut self, dst_id: NodeId, dst_index: usize) -> Box<dyn Any>;

    fn build(
        &mut self,
        sources: &mut Vec<Box<dyn EagerSourceTrait>>,
    );
}

pub(crate) struct EagerSourceBuilder<I, O>
where
    I: FlowIn<I>,
    O: FlowData,
{
    id: SourceId<O>,
    source: Option<Box<dyn SourceFactory<I, O>>>,

    // input: Option<I::Input>,
    // input_meta: Vec<InMeta>,

    output_pipes: Vec<EagerPipe<O>>,
    
    input: Option<I::Input>,
    // output_meta: Vec<OutMeta>,
}

impl<I, O> EagerSourceBuilder<I, O>
where
    I: FlowIn<I>,
    O: FlowData,
{
    fn new(
        id: SourceId<O>,
        source: Box<dyn SourceFactory<I, O>>,
        src_nodes: &I::Nodes,
        builder: &mut impl FlowInBuilder,
    ) -> Self {

        let mut source_infos = Vec::new();
        let input = I::new_input(
            id.id(), 
            src_nodes, 
            &mut source_infos, 
            builder,
        );

        let mut in_arrows = Vec::new();
        I::node_ids(src_nodes, &mut in_arrows);

        Self {
            id: id.clone(),
            source: Some(source),

            input: Some(input),
            // input_meta: source_infos,

            output_pipes: Vec::default(),
            // output_meta: Vec::default(),
        }
    }
}

impl<I, O> EagerSourceBuilderTrait for EagerSourceBuilder<I, O>
where
    I: FlowIn<I>,
    O: FlowData,
{
    unsafe fn add_pipe(&mut self, dst_id: NodeId, input_index: usize) -> Box<dyn Any> {
        let output_index = self.output_pipes.len();

        let pipe = EagerPipe::new(
            self.id.clone(),
            self.output_pipes.len(),
            dst_id,
            input_index
        );

        self.output_pipes.push(pipe.clone());

        let pipe : Box<dyn EagerPipeTrait> = Box::new(pipe);

        unsafe {
            mem::transmute(pipe)
        }
    }

    fn build(
        &mut self,
        sources: &mut Vec<Box<dyn EagerSourceTrait>>,
    ) {
        let source = EagerSource {
            factory: self.source.take().unwrap(),
            source: Box::new(UnsetSource::new()),
            inputs: self.input.take().unwrap(),
            input_pipes: Vec::new(), // Vec<Box<dyn EagerPipeTrait>>,
            output_pipes: self.output_pipes.drain(..).collect(),
    
            is_done: false,
        };

        sources.push(Box::new(source));
    }
}

#[cfg(test)]
mod test {
    use std::{sync::{Arc, Mutex}};

    use source::Source;

    use crate::flow::{
        pipe::{In}, SourceFactory, FlowIn, flow_pool::{self, PoolFlowBuilder}, 
        flow::{Flow, FlowSourcesBuilder},
        FlowOutputBuilder, source::{self, Out}, FlowData, flow_single::EagerFlowBuilder,
    };

    #[test]
    fn test_graph_nil() {
        let builder = EagerFlowBuilder::<()>::new();
        let mut flow = builder.output::<()>(&());

        assert_eq!(flow.call(()), None);
    }

    #[test]
    fn test_graph_node() {
        let vec = Arc::new(Mutex::new(Vec::<String>::default()));
        
        let mut builder = EagerFlowBuilder::<()>::new();
        let ptr = vec.clone();

        let node_id = builder.source::<(), usize>(S::new(move |_: &mut ()| {
            ptr.lock().unwrap().push(format!("Node[]"));
            Ok(Out::None)
        }), &());

        assert_eq!(node_id.index(), 0);

        let mut flow = builder.output::<usize>(&node_id);

        assert_eq!(flow.call(()), None);
        assert_eq!(take(&vec), "Node[]");

        assert_eq!(flow.call(()), None);
        assert_eq!(take(&vec), "Node[]");
    }

    fn take(ptr: &Arc<Mutex<Vec<String>>>) -> String {
        let vec : Vec<String> = ptr.lock().unwrap().drain(..).collect();

        vec.join(", ")
    }

    struct S<I: FlowIn<I>, O: FlowIn<O>> {
        source: Option<Box<dyn Source<I, O>>>,
    }

    impl<I: FlowIn<I>, O: FlowIn<O>> S<I, O> {
        fn new(source: impl Source<I, O>) -> Self {
            Self {
                source: Some(Box::new(source)),
            }
        }
    }

    impl<I: FlowIn<I>, O: FlowIn<O>> SourceFactory<I, O> for S<I, O> {
        fn new(&mut self) -> Box<dyn Source<I, O>> {
            self.source.take().unwrap()
        }
    }
    
    /*
    impl<I, O, F> From<F> for Box<dyn SourceFactory<I, O>>
    where
        I: FlowIn<I> + 'static,
        O: FlowData,
        F: FnMut(&mut I::Input) -> source::Result<Out<O>> + Send + 'static
    {
        fn from(value: F) -> Self {
            let mut item = Some(Box::new(value));
            Box::new(move || item.take().unwrap())
        }
    }
    */
    
    /*
    impl<I, O, F> SourceFactory<I, O> for F
    where
        I: FlowIn<I> + 'static,
        O: FlowData,
        F: FnMut(&mut I::Input) -> source::Result<Out<O>> + Send + 'static
    {
        fn new(&mut self) -> Box<dyn source::Source<I, O>> {
        todo!()
    }
    }
    */
}