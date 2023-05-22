use core::fmt;
use std::{sync::{Mutex, Arc}, marker::PhantomData, cmp, any::Any, mem};

use super::{
    data::{FlowIn},
    dispatch::{InnerWaker, OuterWaker},
    pipe::{Out, PipeOut, PipeIn, pipe}, 
};

pub trait Task<I, O> : Send + 'static
where
    I: FlowIn<I> + 'static,
    O: 'static,
{
    fn init(&mut self) {}
    
    fn next(&mut self, input: &mut I::Input) -> Result<Out<O>>;
}

#[derive(Copy, PartialEq)]
pub struct TaskId<T> {
    index: usize,
    marker: PhantomData<T>,
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct NodeId(usize);

#[derive(Debug)]
pub struct TaskErr;

pub type Result<T> = std::result::Result<T, TaskErr>;

type BoxTask<In, Out> = Box<dyn Task<In, Out>>;

//
// Tasks implementation
//

pub struct Tasks {
    tasks: Vec<TaskNode>,
}

impl Tasks {
    pub(crate) fn init(&mut self) {
        for task in &mut self.tasks {
            task.init();
        }
    }

    pub(crate) fn data_request(
        &self, 
        id: NodeId,
        sink_index: usize,
        n_request: u64,
        waker: &mut dyn OuterWaker,
    ) {
        self.tasks[id.index()].data_request(sink_index, n_request, waker)
    }

    pub(crate) fn data_ready(
        &mut self, 
        id: NodeId,
        source_index: usize,
        n_ready: u64,
        waker: &mut dyn OuterWaker, 
    ) {
        self.tasks[id.index()].data_ready(source_index, n_ready, waker)
    }

    pub(crate) fn complete(
        &self, 
        id: NodeId,
        is_done: bool, 
        waker: &mut dyn OuterWaker,
    ) {
        self.tasks[id.index()].complete(is_done, waker);
    }

    pub(crate) fn execute(
        &mut self, 
        id: NodeId,
        waker: &mut dyn InnerWaker, 
    ) {
        self.tasks[id.index()].execute(waker)
    }
}

impl Default for Tasks {
    fn default() -> Self {
        Self { 
            tasks: Default::default(),
        }
    }
}

//
// TaskNode
//

pub(crate) struct TaskNode {
    _id: NodeId,

    outer: Mutex<Box<dyn TaskOuter>>,
    inner: Mutex<Box<dyn TaskInner>>,
}

impl TaskNode {
    fn new<I, O>(builder: &mut TaskNodeBuilder<I, O>) -> Self
    where
        I: FlowIn<I>,
        O: Send + Clone + 'static
    {
        let input_metas = InMetas(builder.source_info.drain(..).collect());
        let input_metas = Arc::new(Mutex::new(input_metas));

        let out_metas = OutMetas(builder.sink_info.drain(..).collect());
        let out_metas = Arc::new(Mutex::new(out_metas));

        let outer = TaskOuterNode::new(builder, &input_metas, &out_metas);
        let inner = TaskInnerNode::new(builder, &input_metas, &out_metas);

        Self {
            _id: builder.id.id(),
            outer: Mutex::new(Box::new(outer)),
            inner: Mutex::new(Box::new(inner)),
        }
    }

    fn init(&self) {
        self.outer.lock().unwrap().init();
        self.inner.lock().unwrap().init();
    }

    pub(crate) fn data_request(
        &self, 
        out_index: usize,
        n_request: u64,
        waker: &mut dyn OuterWaker,
    ) {
        self.outer.lock().unwrap().data_request(out_index, n_request, waker);
    }

    fn data_ready(
        &mut self, 
        source_index: usize,
        n_ready: u64,
        waker: &mut dyn OuterWaker, 
    ) {
        self.outer.lock().unwrap().data_ready(source_index, n_ready, waker);
    }

    pub(crate) fn complete(
        &self, 
        is_done: bool, 
        waker: &mut dyn OuterWaker,
    ) {
        self.outer.lock().unwrap().post_execute(is_done, waker);
    }

    fn execute(
        &mut self, 
        waker: &mut dyn InnerWaker, 
    ) {
        self.inner.lock().unwrap().next(waker).unwrap();
    }
}

//
// TaskOuter
//

#[derive(Copy, Clone, Debug, PartialEq)]
enum State {
    Idle, // waiting for output request

    Active, // currently dispatched

    Pending, // waiting for input

    Done,
}

trait TaskOuter : Send {
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

pub struct OuterTasks {
    tasks: Vec<Box<dyn TaskOuter>>,
}

impl OuterTasks {
    pub(crate) fn init(&mut self) {
        for task in &mut self.tasks {
            task.init();
        }
    }

    pub(crate) fn data_request(
        &mut self, 
        id: NodeId,
        out_index: usize,
        n_request: u64,
        waker: &mut dyn OuterWaker,
    ) {
        self.tasks[id.index()].data_request(out_index, n_request, waker)
    }

    pub(crate) fn data_ready(
        &mut self, 
        id: NodeId,
        input_index: usize,
        n_ready: u64,
        waker: &mut dyn OuterWaker, 
    ) {
        self.tasks[id.index()].data_ready(input_index, n_ready, waker)
    }

    pub(crate) fn next_complete(
        &mut self, 
        id: NodeId,
        is_done: bool, 
        waker: &mut dyn OuterWaker,
    ) {
        self.tasks[id.index()].post_execute(is_done, waker);
    }
}

struct TaskOuterNode<O> {
    id: TaskId<O>,

    state: State,

    input_meta: Arc<Mutex<InMetas>>,
    output_meta: Arc<Mutex<OutMetas>>,
}

impl<O> TaskOuterNode<O>
where
    O: Clone + 'static
{
    fn new<I>(
        builder: &mut TaskNodeBuilder<I, O>,
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

impl<O> TaskOuter for TaskOuterNode<O>
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
// TaskInner
//

trait TaskInner : Send {
    fn init(&mut self);

    fn next(&mut self, waker: &mut dyn InnerWaker) -> Result<Out<()>>;
}

pub struct InnerTasks {
    tasks: Vec<Mutex<Box<dyn TaskInner>>>,
}

impl InnerTasks {
    pub(crate) fn init(&mut self) {
        for task in &mut self.tasks {
            task.lock().unwrap().init();
        }
    }

    pub(crate) fn next(
        &mut self, 
        id: NodeId,
        waker: &mut dyn InnerWaker,
    ) {
        // TODO: check output
        self.tasks[id.index()].lock().unwrap().next(waker).unwrap();
    }
}

struct TaskInnerNode<I, O>
where
    I: FlowIn<I>
{
    id: TaskId<O>,
    task: BoxTask<I, O>,

    input: I::Input,

    outputs: Vec<Box<dyn PipeOut<O>>>,
    out_index: u32, // round robin index

    input_meta: Arc<Mutex<InMetas>>,
    output_meta: Arc<Mutex<OutMetas>>,
}

impl<I, O> TaskInnerNode<I, O>
where
    I: FlowIn<I>,
    O: Clone + 'static
{
    fn new(
        builder: &mut TaskNodeBuilder<I, O>,
        input_meta: &Arc<Mutex<InMetas>>,
        output_meta: &Arc<Mutex<OutMetas>>,
    ) -> Self {
        Self {
            id: builder.id.clone(), 
            task: builder.task.take().unwrap(),
            input: builder.source.take().unwrap(),

            outputs: builder.sinks.drain(..).collect(),

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

impl<I, O> TaskInner for TaskInnerNode<I, O>
where
    I: FlowIn<I>,
    O: Send + Clone + 'static
{
    fn init(&mut self) {
        self.task.init();

        I::init_source(&mut self.input);

        self.input_meta.lock().unwrap().init();
        self.output_meta.lock().unwrap().init();
    }

    fn next(&mut self, waker: &mut dyn InnerWaker) -> Result<Out<()>> {
        while self.input_meta.lock().unwrap().fill_data::<I>(&mut self.input, waker) {
            match self.task.next(&mut self.input) {
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
// SourceInfo
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

        I::fill_source(input, &mut self.0, &mut index, waker)
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
// SinkInfo
//

struct OutMetas(Vec<PipeOutInfo>);

struct PipeOutInfo {
    dst_id: NodeId,
    input_index: usize,

    n_request: u64,
    n_sent: u64,
}

impl OutMetas {
    fn init(&mut self) {
        for info in &mut self.0 {
            info.init();
        }
    }
    
    fn is_available(&mut self, index: usize) -> bool {
        self.0[index].is_available()
    }

    fn is_any_out_available(&mut self) -> bool {
        for out in &self.0 {
            if out.is_available() {
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

impl PipeOutInfo {
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

impl<T> fmt::Debug for TaskId<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TaskId[{}]", self.index)
    }
}

impl<T> Clone for TaskId<T> {
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

impl<T: 'static> TaskId<T> {
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
// Tasks builder
//

pub struct TasksBuilder {
    tasks: Vec<Box<dyn TaskNodeBuilderTrait>>,
}

pub(crate) trait TaskNodeBuilderTrait {
    //unsafe fn add_sink(&mut self, dst_id: NodeId, dst_index: usize) -> UnsafePtr;
    unsafe fn add_sink(&mut self, dst_id: NodeId, dst_index: usize) -> Box<dyn Any>;

    fn build(&mut self) -> TaskNode;
}

pub(crate) struct TaskNodeBuilder<I, O>
where
    I: FlowIn<I>,
    O: Clone + 'static
{
    id: TaskId<O>,
    task: Option<BoxTask<I, O>>,

    source: Option<I::Input>,
    source_info: Vec<InMeta>,

    sinks: Vec<Box<dyn PipeOut<O>>>,
    sink_info: Vec<PipeOutInfo>,
}

impl TasksBuilder {
    pub(crate) fn new() -> Self {
        Self {
            tasks: Vec::default(),
        }
    }

    pub(crate) fn push_task<I, O>(
        &mut self, 
        task: BoxTask<I, O>,
        src_nodes: &I::Nodes,
    ) -> TaskId<O>
    where
        I: FlowIn<I>,
        O: Send + Clone + 'static
    {
        let id = TaskId::<O>::new(self.tasks.len());
        let task_item = TaskNodeBuilder::new(id.clone(), task, src_nodes, self);
    
        assert_eq!(id.index(), self.tasks.len());

        self.tasks.push(Box::new(task_item));

        id
    }

    pub(crate) fn add_pipe<O: 'static>(
        &mut self,
        src_id: TaskId<O>,
        dst_id: NodeId,
        dst_index: usize,
    ) -> Box<dyn PipeIn<O>> {
        let task = &mut self.tasks[src_id.index()];

        unsafe { 
            // task.add_sink(dst_id, dst_index).downcast::<dyn SourceTrait<O>>().unwrap()

            mem::transmute(task.add_sink(dst_id, dst_index))
        }
    }

    pub(crate) fn build(mut self) -> Tasks {
        let mut tasks = Vec::new();

        for mut task in self.tasks.drain(..) {
            tasks.push(task.build());
        }

        Tasks {
            tasks
        }
    }
}

//
// TaskNode builder
//

impl<I, O> TaskNodeBuilder<I, O>
where
    I: FlowIn<I>,
    O: Clone + 'static
{
    fn new(
        id: TaskId<O>,
        task: Box<dyn Task<I, O>>,
        src_nodes: &I::Nodes,
        tasks: &mut TasksBuilder,
    ) -> Self {
        let mut source_infos = Vec::new();
        let source = I::new_input(
            id.id(), 
            src_nodes, 
            &mut source_infos, 
            tasks,
        );

        let mut in_arrows = Vec::new();
        I::node_ids(src_nodes, &mut in_arrows);

        Self {
            id: id.clone(),
            task: Some(task),

            source: Some(source),
            source_info: source_infos,

            sinks: Vec::default(),
            sink_info: Vec::default(),
        }
    }
}

impl<I, O> TaskNodeBuilderTrait for TaskNodeBuilder<I, O>
where
    I: FlowIn<I>,
    O: Send + Clone + 'static
{
    unsafe fn add_sink(&mut self, dst_id: NodeId, source_index: usize) -> Box<dyn Any> {
        let sink_index = self.sinks.len();

        let (source, sink) = 
            pipe(self.id.clone(), sink_index, dst_id, source_index);

        self.sinks.push(sink);
        self.sink_info.push(PipeOutInfo::new(dst_id, source_index));

        unsafe {
            mem::transmute(source)
        }
    }

    fn build(&mut self) -> TaskNode {
        TaskNode::new(self)
    }
}

//
// Nil task
//

pub struct NilTask;

impl Task<(), ()> for NilTask {
    fn next(&mut self, _source: &mut ()) -> Result<Out<()>> {
        todo!()
    }
}

//
// Function task
//

impl<I, O, F> Task<I, O> for F
where
    I: FlowIn<I> + 'static,
    O: 'static,
    F: FnMut(&mut I::Input) -> Result<Out<O>> + Send + 'static
{
    fn next(&mut self, source: &mut I::Input) -> Result<Out<O>> {
        self(source)
    }
}
