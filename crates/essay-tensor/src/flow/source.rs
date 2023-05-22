use core::fmt;
use std::{sync::{Mutex, Arc}, marker::PhantomData, cmp, any::Any, mem};

use super::{
    data::{FlowIn},
    dispatch::{InnerWaker, OuterWaker},
    pipe::{Out, PipeOut, PipeIn, pipe}, 
};

pub trait Source<I, O> : Send + 'static
where
    I: FlowIn<I> + 'static,
    O: 'static,
{
    fn init(&mut self) {}
    
    fn next(&mut self, input: &mut I::Input) -> Result<Out<O>>;
}

#[derive(Copy, PartialEq)]
pub struct SourceId<T> {
    index: usize,
    marker: PhantomData<T>,
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct NodeId(usize);

#[derive(Debug)]
pub struct SourceErr;

pub type Result<T> = std::result::Result<T, SourceErr>;

type BoxSource<In, Out> = Box<dyn Source<In, Out>>;

//
// Sources implementation
//

pub struct Sources(pub SourcesOuter, pub SourcesInner);

//
// SourceOuter
//

#[derive(Copy, Clone, Debug, PartialEq)]
enum State {
    Idle, // waiting for output request

    Active, // currently dispatched

    Pending, // waiting for input

    Done,
}

pub struct SourcesOuter {
    sources: Vec<Box<dyn SourceOuter>>,
}

impl SourcesOuter {
    pub(crate) fn init(&mut self) {
        for source in &mut self.sources {
            source.init();
        }
    }

    pub(crate) fn data_request(
        &mut self, 
        id: NodeId,
        out_index: usize,
        n_request: u64,
        waker: &mut dyn OuterWaker,
    ) {
        self.sources[id.index()].data_request(out_index, n_request, waker)
    }

    pub(crate) fn data_ready(
        &mut self, 
        id: NodeId,
        input_index: usize,
        n_ready: u64,
        waker: &mut dyn OuterWaker, 
    ) {
        self.sources[id.index()].data_ready(input_index, n_ready, waker)
    }

    pub(crate) fn post_execute(
        &mut self, 
        id: NodeId,
        is_done: bool, 
        waker: &mut dyn OuterWaker,
    ) {
        self.sources[id.index()].post_execute(is_done, waker);
    }
}

pub trait SourceOuter : Send {
    fn init(&mut self);

    fn data_request(
        &mut self, 
        out_index: usize,
        n_request: u64,
        waker: &mut dyn OuterWaker,
    );

    fn data_ready(
        &mut self, 
        input_index: usize,
        n_ready: u64,
        waker: &mut dyn OuterWaker
    );

    fn post_execute(
        &mut self, 
        is_done: bool,
        waker: &mut dyn OuterWaker,
    );
}

struct SourceOuterImpl<O> {
    id: SourceId<O>,

    state: State,

    input_meta: Arc<Mutex<InMetas>>,
    output_meta: Arc<Mutex<OutMetas>>,
}

impl<O> SourceOuterImpl<O>
where
    O: Clone + 'static
{
    fn new<I>(
        builder: &mut SourceBuilder<I, O>,
        input_meta: &Arc<Mutex<InMetas>>,
        output_meta: &Arc<Mutex<OutMetas>>,
    ) -> Self
    where
        I: FlowIn<I>
    {
        Self {
            id: builder.id.clone(),
            state: State::Idle,
            input_meta: input_meta.clone(),
            output_meta: output_meta.clone(),
        }
    }
}

impl<O> SourceOuter for SourceOuterImpl<O>
where
    O: Send + Clone + 'static
{
    fn init(
        &mut self, 
    ) {
        self.state = State::Idle;
    }

    fn data_request(
        &mut self, 
        out_index: usize,
        n_request: u64,
        waker: &mut dyn OuterWaker,
    ) {
        if ! self.output_meta.lock().unwrap().data_request(out_index, n_request) {
            return;
        }

        match self.state {
            State::Idle => {

                if self.input_meta.lock().unwrap().data_request(waker) {
                    self.state = State::Active;

                    waker.execute(self.id.id());
                } else {
                    self.state = State::Pending;
                }
            },
            State::Active => {
            },
            State::Pending => {
            },
            State::Done => {
            },
        }
    }

    fn data_ready(
        &mut self, 
        input_index: usize,
        n_ready: u64,
        waker: &mut dyn OuterWaker
    ) {
        self.input_meta.lock().unwrap().data_ready(input_index, n_ready);

        if let State::Pending = self.state {
            if self.input_meta.lock().unwrap().data_request(waker) {
                self.state = State::Active;
                waker.execute(self.id.id());
            }
        }
    }

    fn post_execute(
        &mut self, 
        is_done: bool,
        waker: &mut dyn OuterWaker,
    ) {
        match self.state {
            State::Active => {
                if is_done {
                    self.state = State::Done;
                } else if ! self.output_meta.lock().unwrap().is_any_out_available() {
                    self.state = State::Idle;
                } else if self.input_meta.lock().unwrap().data_request(waker) {
                    self.state = State::Active;
                    waker.execute(self.id.id());
                } else {
                    self.state = State::Pending;
                }
            },
            _ => { panic!("unexpected state {:?}", self.state) }
        }
    }
}

//
// SourceInner
//

pub trait SourceInner : Send {
    fn init(&mut self);

    fn execute(&mut self, waker: &mut dyn InnerWaker) -> Result<Out<()>>;
}

pub struct SourcesInner {
    sources: Vec<Mutex<Box<dyn SourceInner>>>,
}

impl SourcesInner {
    pub(crate) fn init(&mut self) {
        for source in &mut self.sources {
            source.lock().unwrap().init();
        }
    }

    pub(crate) fn execute(
        &self, 
        id: NodeId,
        waker: &mut dyn InnerWaker,
    ) {
        // TODO: check output
        self.sources[id.index()].lock().unwrap().execute(waker).unwrap();
    }
}

struct SourceInnerImpl<I, O>
where
    I: FlowIn<I>
{
    id: SourceId<O>,
    source: BoxSource<I, O>,

    input: I::Input,

    outputs: Vec<Box<dyn PipeOut<O>>>,
    out_index: u32, // round robin index

    input_meta: Arc<Mutex<InMetas>>,
    output_meta: Arc<Mutex<OutMetas>>,
}

impl<I, O> SourceInnerImpl<I, O>
where
    I: FlowIn<I>,
    O: Clone + 'static
{
    fn new(
        builder: &mut SourceBuilder<I, O>,
        input_meta: &Arc<Mutex<InMetas>>,
        output_meta: &Arc<Mutex<OutMetas>>,
    ) -> Self {
        Self {
            id: builder.id.clone(), 
            source: builder.source.take().unwrap(),
            input: builder.input.take().unwrap(),

            outputs: builder.outputs.drain(..).collect(),

            out_index: 0,

            input_meta: input_meta.clone(),
            output_meta: output_meta.clone(),
        }
    }

    fn next_index(&mut self) -> usize {
        self.out_index = (self.out_index + 1) % self.outputs.len() as u32;

        self.out_index as usize
    }

    fn send(&mut self, value: Option<O>, waker: &mut dyn InnerWaker) -> bool {
        if self.outputs.len() == 0 {
            return false;
        }

        let index = self.out_index as usize;

        self.send_index(index, value, waker);

        let next_index = self.next_index();
        self.output_meta.lock().unwrap().is_available(next_index)
    }

    fn send_all(&mut self, value: Option<O>, waker: &mut dyn InnerWaker) {
        let len = self.outputs.len();

        if len == 0 {
            return;
        }

        for i in 0..len - 1 {
            self.send_index(i, value.clone(), waker);
        }

        self.send_index(len - 1, value, waker);
    }

    fn send_index(
        &mut self, 
        index: usize,
        value: Option<O>,
        waker: &mut dyn InnerWaker,
    ) {
        self.outputs[index].send(value);
        self.output_meta.lock().unwrap().send(index, waker);
    }
} 

impl<I, O> SourceInner for SourceInnerImpl<I, O>
where
    I: FlowIn<I>,
    O: Send + Clone + 'static
{
    fn init(&mut self) {
        self.source.init();

        I::init(&mut self.input);

        self.input_meta.lock().unwrap().init();
        self.output_meta.lock().unwrap().init();
    }

    fn execute(&mut self, waker: &mut dyn InnerWaker) -> Result<Out<()>> {
        while self.input_meta.lock().unwrap().fill_data::<I>(&mut self.input, waker) {
            match self.source.next(&mut self.input) {
                Ok(Out::Some(value)) => {
                    if ! self.send(Some(value), waker) {
                        waker.post_execute(self.id.id(), false);
                        return Ok(Out::Some(()))
                    }
                },
                Ok(Out::None) => {
                    self.send_all(None, waker);
                    waker.post_execute(self.id.id(), true);
                    return Ok(Out::None);
                },
                Ok(Out::Pending) => {
                    waker.post_execute(self.id.id(), false);
                    return Ok(Out::Pending);
                },
                Err(err) => {
                    panic!("Error from task {:?}", err);
                    // return Ok(Out::Pending); // Err(SourceErr::Pending)
                },
            }
        }

        waker.post_execute(self.id.id(), false);
        Ok(Out::Pending)
    }
}

//
// pipe input meta
//

pub struct InMetas(Vec<InMeta>);

pub struct InMeta {
    src_id: NodeId,
    src_out_index: usize, // index of this pipe for the source's output

    request_window: u32,

    n_request: u64,
    n_ready: u64,
    n_read: u64,
}

impl InMetas {
    fn init(&mut self) {
        for meta in &mut self.0 {
            meta.init();
        }
    }

    fn data_request(&mut self, waker: &mut dyn OuterWaker) -> bool {
        let mut is_ready = true;

        for meta in &mut self.0 {
            if ! meta.data_request(waker) {
                is_ready = false;
            }
        }

        is_ready
    }

    fn data_ready(&mut self, index: usize, n_ready: u64) {
        // TODO: cleanup this special case for tail task
        if self.0.len() > 0 {
            let meta = &mut self.0[index];

            meta.n_ready = cmp::max(meta.n_ready, n_ready);
        }
    }

    fn fill_data<I: FlowIn<I>>(
        &mut self, 
        input: &mut I::Input, 
        waker: &mut dyn InnerWaker
    ) -> bool {
        let mut index = 0;

        I::fill(input, &mut self.0, &mut index, waker)
    }
}

impl InMeta {
    pub(crate) fn new(src_id: NodeId, src_index: usize) -> Self {
        Self {
            src_id,
            src_out_index: src_index,

            request_window: 1,
            n_request: 0,
            n_ready: 0,
            n_read: 0,
        }
    }

    fn init(&mut self) {
        self.n_request = 0;
        self.n_ready = 0;
        self.n_read = 0;
    }

    fn data_request(&mut self, waker: &mut dyn OuterWaker) -> bool {
        let mut is_ready = true;

        let n_request = self.n_read + self.request_window as u64;

        if self.n_request < n_request {
            self.n_request = n_request;
            waker.data_request(self.src_id, self.src_out_index, n_request);
            is_ready = false;
        }

        is_ready
    }

    pub(crate) fn set_n_read(&mut self, n_read: u64) {
        self.n_read = n_read;
    }
}

//
// OutMeta
//

struct OutMetas(Vec<OutMeta>);

struct OutMeta {
    dst_id: NodeId,
    input_index: usize,

    n_request: u64,
    n_sent: u64,
}

impl OutMetas {
    fn init(&mut self) {
        for meta in &mut self.0 {
            meta.init();
        }
    }
    
    fn is_available(&mut self, index: usize) -> bool {
        self.0[index].is_available()
    }

    fn is_any_out_available(&mut self) -> bool {
        for meta in &self.0 {
            if meta.is_available() {
                return true;
            }
        }

        return false;
    }

    fn send(&mut self, index: usize, waker: &mut dyn InnerWaker) {
        self.0[index].send(waker);
    }

    fn data_request(&mut self, index: usize, n_request: u64) -> bool {
        self.0[index].data_request(n_request)
    }
}

impl OutMeta {
    pub(crate) fn new(dst_id: NodeId, input_index: usize) -> Self {
        Self {
            dst_id,
            input_index,

            n_request: 0,
            n_sent: 0,
        }
    }

    fn init(&mut self) {
        self.n_request = 0;
        self.n_sent = 0;
    }

    fn is_available(&self) -> bool {
        self.n_sent < self.n_request
    }

    fn send(&mut self, waker: &mut dyn InnerWaker) {
        self.n_sent += 1;
        waker.data_ready(self.dst_id, self.input_index, self.n_sent);
    }

    fn data_request(&mut self, n_request: u64) -> bool {
        if self.n_request < n_request {
            self.n_request = n_request;
            true
        } else {
            false
        }
    }
}

impl<T> fmt::Debug for SourceId<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TaskId[{}]", self.index)
    }
}

impl<T> Clone for SourceId<T> {
    fn clone(&self) -> Self {
        Self { 
            index: self.index,
            marker: self.marker.clone() 
        }
    }
}

impl NodeId {
    pub fn index(&self) -> usize {
        self.0
    }
}

impl<T: 'static> SourceId<T> {
    fn new(index: usize) -> Self {
        Self {
            index,
            marker: PhantomData,
        }
    }

    #[inline]
    pub fn id(&self) -> NodeId {
        NodeId(self.index)
    }

    #[inline]
    pub fn index(&self) -> usize {
        self.index
    }
}

//
// Source builder
//

pub struct SourcesBuilder {
    sources: Vec<Box<dyn SourceBuilderTrait>>,
}

impl SourcesBuilder {
    pub(crate) fn new() -> Self {
        Self {
            sources: Vec::default(),
        }
    }

    pub(crate) fn push_source<I, O>(
        &mut self, 
        source: BoxSource<I, O>,
        in_nodes: &I::Nodes,
    ) -> SourceId<O>
    where
        I: FlowIn<I>,
        O: Send + Clone + 'static
    {
        let id = SourceId::<O>::new(self.sources.len());
        let source = SourceBuilder::new(id.clone(), source, in_nodes, self);
    
        assert_eq!(id.index(), self.sources.len());

        self.sources.push(Box::new(source));

        id
    }

    pub(crate) fn add_pipe<O: 'static>(
        &mut self,
        src_id: SourceId<O>,
        dst_id: NodeId,
        dst_index: usize,
    ) -> Box<dyn PipeIn<O>> {
        let source = &mut self.sources[src_id.index()];

        unsafe { 
            // task.add_sink(dst_id, dst_index).downcast::<dyn SourceTrait<O>>().unwrap()

            mem::transmute(source.add_pipe(dst_id, dst_index))
        }
    }

    pub(crate) fn build(mut self) -> Sources {
        let mut sources_outer: Vec<Box<dyn SourceOuter>> = Vec::new();
        let mut sources_inner: Vec<Mutex<Box<dyn SourceInner>>> = Vec::new();

        for mut source in self.sources.drain(..) {
            //tasks.push(task.build());
            source.build(&mut sources_outer, &mut sources_inner);
        }

        let sources_outer = SourcesOuter {
            sources: sources_outer,
        };

        let sources_inner = SourcesInner {
            sources: sources_inner,
        };
        
        Sources(sources_outer, sources_inner)
    }
}

//
// TaskNode builder
//

pub(crate) trait SourceBuilderTrait {
    //unsafe fn add_sink(&mut self, dst_id: NodeId, dst_index: usize) -> UnsafePtr;
    unsafe fn add_pipe(&mut self, dst_id: NodeId, dst_index: usize) -> Box<dyn Any>;

    fn build(
        &mut self,
        sources_outer: &mut Vec<Box<dyn SourceOuter>>,
        sources_inner: &mut Vec<Mutex<Box<dyn SourceInner>>>,
    );
}

pub(crate) struct SourceBuilder<I, O>
where
    I: FlowIn<I>,
    O: Clone + 'static
{
    id: SourceId<O>,
    source: Option<BoxSource<I, O>>,

    input: Option<I::Input>,
    input_meta: Vec<InMeta>,

    outputs: Vec<Box<dyn PipeOut<O>>>,
    output_meta: Vec<OutMeta>,
}

impl<I, O> SourceBuilder<I, O>
where
    I: FlowIn<I>,
    O: Clone + 'static
{
    fn new(
        id: SourceId<O>,
        source: Box<dyn Source<I, O>>,
        src_nodes: &I::Nodes,
        sources: &mut SourcesBuilder,
    ) -> Self {
        let mut source_infos = Vec::new();
        let input = I::new_input(
            id.id(), 
            src_nodes, 
            &mut source_infos, 
            sources,
        );

        let mut in_arrows = Vec::new();
        I::node_ids(src_nodes, &mut in_arrows);

        Self {
            id: id.clone(),
            source: Some(source),

            input: Some(input),
            input_meta: source_infos,

            outputs: Vec::default(),
            output_meta: Vec::default(),
        }
    }
}

impl<I, O> SourceBuilderTrait for SourceBuilder<I, O>
where
    I: FlowIn<I>,
    O: Send + Clone + 'static
{
    unsafe fn add_pipe(&mut self, dst_id: NodeId, input_index: usize) -> Box<dyn Any> {
        let output_index = self.outputs.len();

        let (input, output) = 
            pipe(self.id.clone(), output_index, dst_id, input_index);

        self.outputs.push(output);
        self.output_meta.push(OutMeta::new(dst_id, input_index));

        unsafe {
            mem::transmute(input)
        }
    }

    fn build(
        &mut self,
        sources_outer: &mut Vec<Box<dyn SourceOuter>>,
        sources_inner: &mut Vec<Mutex<Box<dyn SourceInner>>>,
    ) {
        let input_meta = InMetas(self.input_meta.drain(..).collect());
        let input_meta = Arc::new(Mutex::new(input_meta));

        let output_meta = OutMetas(self.output_meta.drain(..).collect());
        let output_meta = Arc::new(Mutex::new(output_meta));

        let outer = SourceOuterImpl::new(self, &input_meta, &output_meta);
        let inner = SourceInnerImpl::new(self, &input_meta, &output_meta);

        sources_outer.push(Box::new(outer));
        sources_inner.push(Mutex::new(Box::new(inner)));
    }
}

//
// Nil task
//

pub struct NilTask;

impl Source<(), ()> for NilTask {
    fn next(&mut self, _source: &mut ()) -> Result<Out<()>> {
        todo!()
    }
}

//
// Function Source
//

impl<I, O, F> Source<I, O> for F
where
    I: FlowIn<I> + 'static,
    O: 'static,
    F: FnMut(&mut I::Input) -> Result<Out<O>> + Send + 'static
{
    fn next(&mut self, input: &mut I::Input) -> Result<Out<O>> {
        self(input)
    }
}
