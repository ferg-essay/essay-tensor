use std::{
    any::Any,
    mem, 
    marker::PhantomData, cell::RefCell, collections::HashMap, 
};

use crate::flow::source::NoneSourceFactory;

use super::{
    FlowData, FlowIn, Source, 
    Out, pipe::{PipeIn, PipeSingleTrait, PipeSingle}, SourceId, SourceFactory, FlowOutputBuilder, flow::{Flow, FlowSourcesBuilder}, source::{NodeId, SharedOutput, OutputSource, TailSource, NoneSource}, data::FlowInBuilder,
};


#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PipeId(usize);

pub struct FlowSingle<I, O>
where
    I: FlowIn<I>,
    O: FlowIn<O>,
{
    sources: Vec<RefCell<Box<dyn SourceTraitSingle>>>,

    _input: I::Nodes,
    last_id: SourceId<O>,

    output: SharedOutput<O>,
    tail_id: SourceId<bool>,

}

impl<I, O> FlowSingle<I, O>
where
    I: FlowIn<I>,
    O: FlowIn<O>,
{
    fn next(&mut self, _count: u64) -> Option<O> {
        self.sources[self.tail_id.index()].borrow_mut().execute(self);

        self.output.take()
    }

    pub(crate) fn export(mut self, builder: &mut FlowInBuilderSingle) -> SourceId<O> {
        let mut node_map : HashMap<NodeId, NodeId> = HashMap::new();

        for source in self.sources.drain(..) {
            let mut source = source.into_inner();

            source.export(&mut node_map, builder);
        }

        SourceId::from(node_map.get(&self.last_id.id()).unwrap())
    }
}

impl<I, O> Flow<I, O> for FlowSingle<I, O>
where
    I: FlowIn<I>,
    O: FlowIn<O>,
{
    type Iter<'a> = FlowIterSingle<'a, I, O>;

    fn iter<'a>(&'a mut self, _input: I) -> Self::Iter<'a> {
        self.output.take();

        for source in &self.sources {
            source.borrow_mut().init();
        }

        FlowIterSingle {
            flow: self,
            count: 0,
            marker: PhantomData,
        }
    }
}

pub trait FlowSingleTrait {
    fn data_request(&self, src_id: NodeId, src_index: usize, n_request: u64);
}

impl<I, O> FlowSingleTrait for FlowSingle<I, O>
where
    I: FlowIn<I>,
    O: FlowIn<O>
{
    fn data_request(&self, src_id: NodeId, src_index: usize, n_request: u64) {
        self.sources[src_id.index()].borrow_mut().data_request(src_index, n_request, self);
    }
}

//
// Flow iterator
//

pub struct FlowIterSingle<'a, I: FlowIn<I>, O: FlowIn<O>> {
    flow: &'a mut FlowSingle<I, O>,
    count: u64,
    marker: PhantomData<O>,
}

impl<I: FlowIn<I>, O: FlowIn<O>> Iterator for FlowIterSingle<'_, I, O> {
    type Item = O;

    fn next(&mut self) -> Option<Self::Item> {
        let count = self.count + 1;
        self.count = count;

        self.flow.next(count)
    }
}

//
// Source 
// 

pub struct SourceSingle<I: FlowIn<I>, O: FlowData>
where
    I: FlowIn<I>,
    O: FlowIn<O>
{
    id: SourceId<O>,
    factory: Box<dyn SourceFactory<I, O>>,
    source: Box<dyn Source<I, O>>,
    inputs: I::Input,
    input_pipes: Vec<Box<dyn PipeSingleTrait>>,
    output_pipes: Vec<PipeSingle<O>>,

    _is_done: bool,
}

impl<I, O> SourceSingle<I, O>
where
    I: FlowIn<I>,
    O: FlowData
{
    fn input_request(&mut self, flow: &dyn FlowSingleTrait) {
        for pipe in &self.input_pipes {
            if pipe.is_pending() {
                flow.data_request(pipe.src_id(), pipe.src_index(), pipe.n_read() + 1);
            }
        }
    }
}

pub trait SourceTraitSingle {
    fn init(&mut self);

    fn data_request(&mut self, out_index: usize, n_request: u64, flow: &dyn FlowSingleTrait);

    fn execute(&mut self, flow: &dyn FlowSingleTrait);

    fn export(
        &mut self, 
        node_map: &mut HashMap<NodeId, NodeId>, 
        builder: &mut FlowInBuilderSingle);
}

impl<I, O> SourceTraitSingle for SourceSingle<I, O>
where
    I: FlowIn<I>,
    O: FlowData
{
    fn init(&mut self) {
        self.source = self.factory.new();

        for pipe in &mut self.output_pipes {
            pipe.init();
        }

        I::init(&mut self.inputs);
    }

    fn data_request(&mut self, out_index: usize, n_request: u64, flow: &dyn FlowSingleTrait) {
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

    fn execute(&mut self, flow: &dyn FlowSingleTrait) {
        self.input_request(flow);

        match self.source.next(&mut self.inputs) {
            Ok(Out::Some(_)) => {
                return;
            }
            Ok(Out::None) => {
                return;
            }
            Ok(Out::Pending) => {
                panic!("Unexpected pending");
            }
            Err(err) => { panic!("Unknown error {:?}", &err); }
        }
    }

    fn export(
        &mut self, 
        node_map: &mut HashMap<NodeId, NodeId>, 
        builder: &mut FlowInBuilderSingle
    ) {
        let none_factory : Box<dyn SourceFactory<I, O>> = Box::new(NoneSourceFactory::new());

        let factory = mem::replace(&mut self.factory, none_factory);

        let in_nodes = I::export(&mut self.inputs, node_map);
        
        let id = builder.add_factory(factory, &in_nodes);

        node_map.insert(self.id.id(), id.id());
    }
}

//
// FlowBuilder single builder
//

pub struct FlowBuilderSingle<I: FlowIn<I>> {
    inputs: I::Nodes,
    
    sources: FlowInBuilderSingle, // Vec<Box<dyn SourceBuilderSingleTrait>>,
}

impl<I: FlowIn<I>> FlowBuilderSingle<I> {
    pub(crate) fn new() -> Self {
        let mut sources_builder = FlowInBuilderSingle {
            sources: Vec::new(),
        };

        let inputs = I::new_flow_input(&mut sources_builder);

        Self {
            inputs,
            sources: sources_builder,
        }
    }

    pub(crate) fn sources(&mut self) -> &mut FlowInBuilderSingle {
        &mut self.sources
    }
}

impl<In: FlowIn<In>> FlowSourcesBuilder for FlowBuilderSingle<In> {
    fn source<I, O>(
        &mut self, 
        source: impl SourceFactory<I, O>,
        in_nodes: &I::Nodes,
    ) -> SourceId<O>
    where
        I: FlowIn<I>,
        O: FlowData,
    {
        self.sources.add_source(source, in_nodes)
    }
}

impl<In: FlowIn<In>> FlowOutputBuilder<In> for FlowBuilderSingle<In> {
    type Flow<O: FlowData> = FlowSingle<In, O>;
    
    fn output<O: FlowData>(mut self, src_id: &SourceId<O>) -> Self::Flow<O> {
        let shared_output = SharedOutput::<O>::new();

        let output_ptr = shared_output.clone();

        let id = self.source::<O, bool>(move || {
            OutputSource::new(output_ptr.clone())
        }, src_id);
        let _tail = self.source::<bool, bool>(|| TailSource, &id);

        let mut flow_sources = Vec::new();

        for mut source in self.sources.sources.drain(..) {
            source.build(&mut flow_sources)
        }

        FlowSingle {
            sources: flow_sources,

            _input: self.inputs,
            last_id: src_id.clone(),
            output: shared_output, // In::new(source),
            tail_id: id,
        }
    }

    fn input(&mut self) -> &In::Nodes {
        todo!()
    }
}

pub struct FlowInBuilderSingle {
    sources: Vec<Box<dyn SourceBuilderSingleTrait>>,
}

impl FlowInBuilderSingle {
    fn get_pipe(&self, src_id: NodeId, src_index: usize) -> Box<dyn PipeSingleTrait> {
        self.sources[src_id.index()].get_pipe(src_index)
    }

    fn add_factory<I, O>(
        &mut self, 
        factory: Box<dyn SourceFactory<I, O>>, 
        in_nodes: &I::Nodes
    ) -> SourceId<O>
    where 
        I: FlowIn<I>, 
        O: FlowData 
    {
        let id = SourceId::<O>::new(self.sources.len());
        let source = SourceBuilderSingle::new(
            id.clone(), 
            factory,
            in_nodes, 
            self,
        );
    
        assert_eq!(id.index(), self.sources.len());

        self.sources.push(Box::new(source));

        id
    }
}

impl FlowInBuilder for FlowInBuilderSingle {
    fn add_source<I, O>(
        &mut self, 
        source: impl SourceFactory<I, O>,
        in_nodes: &I::Nodes,
    ) -> SourceId<O>
    where
        I: FlowIn<I>,
        O: FlowData,
    {
        let id = SourceId::<O>::new(self.sources.len());
        let source = SourceBuilderSingle::new(
            id.clone(), 
            Box::new(source),
            in_nodes, 
            self,
        );
    
        assert_eq!(id.index(), self.sources.len());

        self.sources.push(Box::new(source));

        id
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
/*
impl<In: FlowIn<In>> FlowInBuilder for FlowBuilderSingle<In> {
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
*/

//
// Eager builder
//

pub(crate) struct SourceBuilderSingle<I, O>
where
    I: FlowIn<I>,
    O: FlowData,
{
    id: SourceId<O>,
    source: Option<Box<dyn SourceFactory<I, O>>>,

    // input: Option<I::Input>,
    // input_meta: Vec<InMeta>,

    input_pipes: Vec<Box<dyn PipeSingleTrait>>,
    output_pipes: Vec<PipeSingle<O>>,

    input: Option<I::Input>,
    // output_meta: Vec<OutMeta>,
}

impl<I, O> SourceBuilderSingle<I, O>
where
    I: FlowIn<I>,
    O: FlowData,
{
    fn new(
        id: SourceId<O>,
        source: Box<dyn SourceFactory<I, O>>,
        src_nodes: &I::Nodes,
        builder: &mut FlowInBuilderSingle,
    ) -> Self {

        let mut source_meta = Vec::new();
        let input = I::new_input(
            id.id(), 
            src_nodes, 
            &mut source_meta, 
            builder,
        );

        let mut input_pipes = Vec::new();

        for meta in &source_meta {
            input_pipes.push(builder.get_pipe(meta.src_id(), meta.src_index()));
        }

        let mut in_arrows = Vec::new();
        I::node_ids(src_nodes, &mut in_arrows);

        Self {
            id: id.clone(),
            source: Some(source),

            input: Some(input),

            input_pipes,

            output_pipes: Vec::default(),
        }
    }
}

pub(crate) trait SourceBuilderSingleTrait {
    unsafe fn add_pipe(&mut self, dst_id: NodeId, dst_index: usize) -> Box<dyn Any>;

    fn get_pipe(&self, src_index: usize) -> Box<dyn PipeSingleTrait>;

    fn build(
        &mut self,
        sources: &mut Vec<RefCell<Box<dyn SourceTraitSingle>>>,
    );
}

impl<I, O> SourceBuilderSingleTrait for SourceBuilderSingle<I, O>
where
    I: FlowIn<I>,
    O: FlowData,
{
    unsafe fn add_pipe(&mut self, dst_id: NodeId, input_index: usize) -> Box<dyn Any> {
        // let output_index = self.output_pipes.len();

        let pipe = PipeSingle::new(
            self.id.clone(),
            self.output_pipes.len(),
            dst_id,
            input_index
        );

        self.output_pipes.push(pipe.clone());

        let pipe : Box<dyn PipeIn<O>> = Box::new(pipe);

        unsafe {
            mem::transmute(pipe)
        }
     }

    fn get_pipe(&self, src_index: usize) -> Box<dyn PipeSingleTrait> {
        Box::new(self.output_pipes[src_index].clone())
    }

    fn build(
        &mut self,
        sources: &mut Vec<RefCell<Box<dyn SourceTraitSingle>>>,
    ) {
        let source = SourceSingle {
            id: self.id.clone(),
            factory: self.source.take().unwrap(),
            source: Box::new(NoneSource::new()),
            inputs: self.input.take().unwrap(),
            input_pipes: self.input_pipes.drain(..).collect(),
            output_pipes: self.output_pipes.drain(..).collect(),
    
            _is_done: false,
        };

        sources.push(RefCell::new(Box::new(source)));
    }
}

#[cfg(test)]
mod test {
    use std::{sync::{Arc, Mutex}};

    use source::Source;

    use crate::flow::{
        pipe::{In}, SourceFactory, FlowIn,
        flow::{Flow, FlowSourcesBuilder},
        FlowOutputBuilder, source::{self, Out}, flow_single::FlowBuilderSingle,
    };

    #[test]
    fn flow_nil() {
        todo!();
        /*
        let builder = FlowBuilderSingle::<()>::new();
        let mut flow = builder.output::<()>(&());

        assert_eq!(flow.call(()), None);
        */
    }

    #[test]
    fn flow_node() {
        let vec = Arc::new(Mutex::new(Vec::<String>::default()));
        
        let mut builder = FlowBuilderSingle::<()>::new();
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

    #[test]
    fn flow_detached_node() {
        let vec = Arc::new(Mutex::new(Vec::<String>::default()));
        
        let mut builder = FlowBuilderSingle::<()>::new();
        let ptr = vec.clone();

        let node_id = builder.source::<(), bool>(S::new(move |_: &mut ()| {
            ptr.lock().unwrap().push(format!("Node[]"));
            Ok(Out::None)
        }), &());

        assert_eq!(node_id.index(), 0);

        todo!();
        /*
        let mut flow = builder.output::<()>(&());

        assert_eq!(flow.call(()), None);
        assert_eq!(take(&vec), "");

        assert_eq!(flow.call(()), None);
        assert_eq!(take(&vec), "");
        */
    }

    #[test]
    fn flow_sequence() {
        let vec = Arc::new(Mutex::new(Vec::<String>::default()));
        
        let mut builder = FlowBuilderSingle::<()>::new();

        let ptr = vec.clone();
        let mut data = vec!["a".to_string(), "b".to_string()];

        let n_0 = builder.source::<(), String>(S::new(move |_: &mut ()| {
            ptr.lock().unwrap().push(format!("Node0[]"));
            match data.pop() {
                Some(v) => {
                    Ok(Out::Some(v))
                },
                None => {
                    Ok(Out::None)
                }
            }
        }), &());

        assert_eq!(n_0.index(), 0);

        let ptr = vec.clone();
        let n_1 = builder.source::<String, bool>(S::new(move |s: &mut In<String>| {
            ptr.lock().unwrap().push(format!("Node1[{:?}]", s.next()));
            Ok(Out::None)
        }), &n_0);

        assert_eq!(n_1.index(), 1);

        let mut flow = builder.output::<bool>(&n_1); // n_1);

        assert_eq!(flow.call(()), None);
        assert_eq!(take(&vec), "Node0[], Node1[Some(\"b\")]");

        assert_eq!(flow.call(()), None);
        assert_eq!(take(&vec), "Node0[], Node1[Some(\"a\")]");

        assert_eq!(flow.call(()), None);
        assert_eq!(take(&vec), "Node0[], Node1[None]");

        assert_eq!(flow.call(()), None);
        assert_eq!(take(&vec), "Node0[], Node1[None]");
    }

    #[test]
    fn flow_iter() {
        let vec = Arc::new(Mutex::new(Vec::<String>::default()));
        
        let mut builder = FlowBuilderSingle::<()>::new();

        let ptr = vec.clone();
        let mut data = vec!["a".to_string(), "b".to_string(), "c".to_string()];

        let n_0 = builder.source::<(), String>(S::new(move |_: &mut ()| {
            ptr.lock().unwrap().push(format!("Node0[]"));
            match data.pop() {
                Some(v) => {
                    Ok(Out::Some(v))
                },
                None => {
                    Ok(Out::None)
                }
            }
        }), &());

        assert_eq!(n_0.index(), 0);

        let ptr = vec.clone();
        let n_1 = builder.source::<String, String>(S::new(move |s: &mut In<String>| {
            let value = s.next();
            ptr.lock().unwrap().push(format!("Node1[{:?}]", &value));
            Ok(Out::from(value))
        }), &n_0);

        assert_eq!(n_1.index(), 1);

        let mut flow = builder.output::<String>(&n_1); // n_1);

        let mut iter = flow.iter(());
        assert_eq!(take(&vec), "");

        assert_eq!(iter.next(), Some("c".to_string()));
        assert_eq!(take(&vec), "Node0[], Node1[Some(\"c\")]");

        assert_eq!(iter.next(), Some("b".to_string()));
        assert_eq!(take(&vec), "Node0[], Node1[Some(\"b\")]");

        assert_eq!(iter.next(), Some("a".to_string()));
        assert_eq!(take(&vec), "Node0[], Node1[Some(\"a\")]");

        assert_eq!(iter.next(), None);
        assert_eq!(take(&vec), "Node0[], Node1[None]");

        assert_eq!(iter.next(), None);
        assert_eq!(take(&vec), "");

        assert_eq!(iter.next(), None);
        assert_eq!(take(&vec), "");
    }

    fn take(ptr: &Arc<Mutex<Vec<String>>>) -> String {
        let vec : Vec<String> = ptr.lock().unwrap().drain(..).collect();

        vec.join(", ")
    }

    #[derive(Clone)]
    struct S<I: FlowIn<I>, O: FlowIn<O>> {
        source: Arc<Mutex<Box<dyn Source<I, O>>>>,
    }

    impl<I: FlowIn<I>, O: FlowIn<O>> Source<I, O> for S<I, O> {
        fn next(&mut self, input: &mut I::Input) -> source::Result<Out<O>> {
            self.source.lock().unwrap().next(input)
        }
    }

    impl<I: FlowIn<I>, O: FlowIn<O>> S<I, O> {
        fn new(source: impl Source<I, O>) -> Self {
            Self {
                source: Arc::new(Mutex::new(Box::new(source))),
            }
        }
    }

    impl<I: FlowIn<I>, O: FlowIn<O>> SourceFactory<I, O> for S<I, O> {
        fn new(&mut self) -> Box<dyn Source<I, O>> {
            Box::new(self.clone())
        }
    }
}