use std::{sync::{Mutex, Arc}};

use super::{
    data::{FlowIn},
    dispatch::{Dispatcher}, graph::{NodeId, TaskId, Graph}, 
    source::{Out, SinkTrait, SourceTrait, task_channel}, 
    ptr::UnsafePtr,
};

#[derive(Debug)]
pub struct TaskErr;

pub type Result<T> = std::result::Result<T, TaskErr>;

pub trait Task<I, O> : Send + 'static
where
    I: FlowIn<I> + 'static,
    O: 'static,
{
    fn init(&mut self) {}
    
    fn call(&mut self, source: &mut I::Source) -> Result<Out<O>>;
}

pub struct Tasks {
    tasks: Vec<TaskNode>,
}

pub(crate) struct TaskNode {
    _id: NodeId,

    outer: Mutex<Box<dyn TaskOuter>>,
    inner: Mutex<Box<dyn TaskInner>>,
}

trait TaskPtr {}

trait TaskOuter {
    fn init(&mut self);

    fn request_source(
        &mut self, 
        src_index: usize,
        n_request: u64,
        waker: &mut Dispatcher,
    );

    fn ready_source(
        &mut self, 
        // data: &mut GraphData, 
        waker: &mut Dispatcher
    );

    fn complete(
        &mut self, 
        is_done: bool,
        waker: &mut Dispatcher,
    );
}

struct TaskOuterNode<O> {
    id: TaskId<O>,

    state: NodeState,

    source_info: Arc<Mutex<SourceInfos>>,
    sink_info: Arc<Mutex<SinkInfos>>,
}

trait TaskInner {
    fn init(&mut self);

    fn wake(
        &mut self, 
        dispatcher: &mut Dispatcher,
    ) -> Out<()>;

    fn execute(&mut self, waker: &mut Dispatcher) -> Result<Out<()>>;
}

struct TaskInnerNode<I, O>
where
    I: FlowIn<I>
{
    id: TaskId<O>,
    task: BoxTask<I, O>,

    source: I::Source,

    sinks: Vec<Box<dyn SinkTrait<O>>>,
    sink_index: u32,

    source_info: Arc<Mutex<SourceInfos>>,
    sink_info: Arc<Mutex<SinkInfos>>,
}

pub struct SourceInfo {
    src_id: NodeId,
    src_index: usize,

    request_window: u32,

    n_read: u64,
    n_request: u64,
}

struct SourceInfos(Vec<SourceInfo>);

struct SinkInfo {
    _dst_id: NodeId,
    n_request: u64,
    n_sent: u64,

    update_ticks: u64,
}

struct SinkInfos(Vec<SinkInfo>);

type BoxTask<In, Out> = Box<dyn Task<In, Out>>;

#[derive(Copy, Clone, Debug, PartialEq)]
enum NodeState {
    Idle, // waiting for output request

    Active, // currently dispatched

    WaitingIn, // waiting for input

    Done,
}

//
// Task builder structs
//

pub struct TasksBuilder {
    tasks: Vec<Box<dyn TaskNodeBuilderTrait>>,
}

pub(crate) trait TaskNodeBuilderTrait {
    unsafe fn add_sink(&mut self, dst_id: NodeId, dst_index: usize) -> UnsafePtr;

    fn build(&mut self) -> TaskNode;
}

pub(crate) struct TaskNodeBuilder<I, O>
where
    I: FlowIn<I>,
    O: Clone + 'static
{
    id: TaskId<O>,
    task: Option<BoxTask<I, O>>,

    source: Option<I::Source>,
    source_info: Vec<SourceInfo>,

    sinks: Vec<Box<dyn SinkTrait<O>>>,
    sink_info: Vec<SinkInfo>,
}

//
// Tasks implementation
//

impl Tasks {
    pub(crate) fn init(&mut self) {
        for task in &mut self.tasks {
            task.init();
        }
    }

    pub(crate) fn request_source(
        &self, 
        id: NodeId,
        src_index: usize,
        n_request: u64,
        waker: &mut Dispatcher,
    ) {
        self.tasks[id.index()].request_source(src_index, n_request, waker)
    }

    pub(crate) fn ready_source(
        &mut self, 
        id: NodeId,
        waker: &mut Dispatcher, 
    ) {
        self.tasks[id.index()].ready_source(waker)
    }

    pub(crate) fn execute(
        &mut self, 
        id: NodeId,
        waker: &mut Dispatcher, 
    ) {
        self.tasks[id.index()].execute(waker)
    }

    pub(crate) fn complete(
        &self, 
        id: NodeId,
        is_done: bool, 
        waker: &mut Dispatcher,
    ) {
        self.tasks[id.index()].complete(is_done, waker);
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

impl TaskNode {
    fn new<I, O>(builder: &mut TaskNodeBuilder<I, O>) -> Self
    where
        I: FlowIn<I>,
        O: Clone + 'static
    {
        let source_info = SourceInfos(builder.source_info.drain(..).collect());
        let source_info = Arc::new(Mutex::new(source_info));

        let sink_info = SinkInfos(builder.sink_info.drain(..).collect());
        let sink_info = Arc::new(Mutex::new(sink_info));

        let outer = TaskOuterNode::new(builder, &source_info, &sink_info);
        let inner = TaskInnerNode::new(builder, &source_info, &sink_info);

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

    pub(crate) fn request_source(
        &self, 
        src_index: usize,
        n_request: u64,
        waker: &mut Dispatcher,
    ) {
        self.outer.lock().unwrap().request_source(src_index, n_request, waker);
    }

    fn ready_source(
        &mut self, 
        waker: &mut Dispatcher, 
    ) {
        self.outer.lock().unwrap().ready_source(waker);
    }

    fn execute(
        &mut self, 
        waker: &mut Dispatcher, 
    ) {
        match self.inner.lock().unwrap().execute(waker) {
            Ok(_) => {
            },
            Err(_) => todo!(),
        }
    }

    pub(crate) fn complete(
        &self, 
        is_done: bool, 
        waker: &mut Dispatcher,
    ) {
        self.outer.lock().unwrap().complete(is_done, waker);
    }
}

//
// TaskOuter
//

impl<O> TaskOuterNode<O>
where
    O: Clone + 'static
{
    fn new<I>(
        builder: &mut TaskNodeBuilder<I, O>,
        source_info: &Arc<Mutex<SourceInfos>>,
        sink_info: &Arc<Mutex<SinkInfos>>,
    ) -> Self
    where
        I: FlowIn<I>
    {
        Self {
            id: builder.id.clone(),
            state: NodeState::Idle,
            // in_nodes: builder.in_nodes.clone(),
            // arrows_out: Vec::default(),
            source_info: source_info.clone(),
            sink_info: sink_info.clone(),
        }
    }
}

impl<O> TaskOuter for TaskOuterNode<O>
where
    O: Clone + 'static
{
    fn init(
        &mut self, 
    ) {
        self.state = NodeState::Idle;
    }

    fn request_source(
        &mut self, 
        sink_index: usize,
        n_request: u64,
        waker: &mut Dispatcher,
    ) {
        if ! self.sink_info.lock().unwrap().request_source(sink_index, n_request) {
            return;
        }

        match self.state {
            NodeState::Idle => {
                self.state = NodeState::WaitingIn;

                if self.source_info.lock().unwrap().request_source(waker) {
                    self.state = NodeState::Active;
                    waker.execute(self.id.id());
                }
            },
            NodeState::Active => {
            },
            NodeState::WaitingIn => {
            },
            NodeState::Done => {
            },
        }
    }

    fn ready_source(
        &mut self, 
        waker: &mut Dispatcher
    ) {
        match self.state {
            NodeState::Idle => { 
                // self.state = NodeState::Active; // WaitingIn;
                // dispatcher.spawn(self.id.id());
            },
            NodeState::Active => {},
            NodeState::WaitingIn => {
                if self.source_info.lock().unwrap().request_source(waker) {
                    self.state = NodeState::Active;
                    waker.execute(self.id.id());
                }
            },
            NodeState::Done => todo!(),
        }
    }

    fn complete(
        &mut self, 
        is_done: bool,
        waker: &mut Dispatcher,
    ) {
        match self.state {
            NodeState::Active => {
                if is_done {
                    self.state = NodeState::Done;
                }
                else if ! self.sink_info.lock().unwrap().is_any_sink_available() {
                    self.state = NodeState::Idle;
                } else if self.source_info.lock().unwrap().request_source(waker) {
                    self.state = NodeState::Active;
                    waker.execute(self.id.id());
                } else {
                    self.state = NodeState::WaitingIn;
                }
            },
            _ => { panic!("unexpected state {:?}", self.state) }
        }
    }
}

//
// TaskInner
//

impl<I, O> TaskInnerNode<I, O>
where
    I: FlowIn<I>,
    O: Clone + 'static
{
    fn new(
        builder: &mut TaskNodeBuilder<I, O>,
        source_info: &Arc<Mutex<SourceInfos>>,
        sink_info: &Arc<Mutex<SinkInfos>>,
    ) -> Self {
        Self {
            id: builder.id.clone(), 
            task: builder.task.take().unwrap(),
            source: builder.source.take().unwrap(),

            sinks: builder.sinks.drain(..).collect(),

            sink_index: 0,
            source_info: source_info.clone(),
            sink_info: sink_info.clone(),
        }
    }

    fn next_index(&mut self) -> usize {
        self.sink_index = (self.sink_index + 1) % self.sinks.len() as u32;

        self.sink_index as usize
    }

    fn send(&mut self, value: Option<O>, waker: &mut Dispatcher) -> bool {
        if self.sinks.len() == 0 {
            return false;
        }

        let index = self.sink_index as usize;

        self.send_index(index, value, waker);

        let next_index = self.next_index();
        self.sink_info.lock().unwrap().is_available(next_index)
    }

    fn send_all(&mut self, value: Option<O>, waker: &mut Dispatcher) {
        let len = self.sinks.len();

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
        waker: &mut Dispatcher,
    ) {
        self.sinks[index].send(value.clone());
        self.sink_info.lock().unwrap().send(index);
        waker.ready_source(self.sinks[index].dst_id());
    }
} 

impl<I, O> TaskInner for TaskInnerNode<I, O>
where
    I: FlowIn<I>,
    O: Clone + 'static
{
    fn init(&mut self) {
        self.task.init();

        I::init_source(&mut self.source);

        self.source_info.lock().unwrap().init();
        self.sink_info.lock().unwrap().init();
    }

    fn wake(&mut self, waker: &mut Dispatcher) -> Out<()> {
        if I::fill_source(&mut self.source, waker) {
            waker.execute(self.id.id());

            Out::Some(())
        } else {
            Out::Pending
        }
    }

    fn execute(&mut self, waker: &mut Dispatcher) -> Result<Out<()>> {
        while I::fill_source(&mut self.source, waker) {
            match self.task.call(&mut self.source) {
                Ok(Out::Some(value)) => {
                    if ! self.send(Some(value), waker) {
                        return Ok(Out::Some(()))
                    }
                },
                Ok(Out::None) => {
                    self.send_all(None, waker);
                    return Ok(Out::None);
                },
                Ok(Out::Pending) => {
                    return Ok(Out::Pending);
                },
                Err(err) => {
                    panic!("Error from task {:?}", err);
                    // return Ok(Out::Pending); // Err(SourceErr::Pending)
                },
            }
        }

        Ok(Out::Pending)
    }
}

//
// SourceInfo
//

impl SourceInfos {
    fn init(&mut self) {
        for info in &mut self.0 {
            info.init();
        }
    }

    fn request_source(&mut self, waker: &mut Dispatcher) -> bool {
        let mut is_ready = true;

        for info in &mut self.0 {
            if ! info.request_source(waker) {
                is_ready = false;
            }
        }

        is_ready
    }
}

impl SourceInfo {
    pub(crate) fn new(src_id: NodeId, src_index: usize) -> Self {
        Self {
            src_id,
            src_index,

            request_window: 1,
            n_read: 0,
            n_request: 0,
        }
    }

    fn init(&mut self) {
        self.n_request = 0;
        self.n_read = 0;
    }

    fn request_source(&mut self, waker: &mut Dispatcher) -> bool {
        let mut is_ready = true;

        let n_request = self.n_read + self.request_window as u64;

        if self.n_request < n_request {
            self.n_request = n_request;
            waker.request_source(self.src_id, self.src_index, n_request);
            is_ready = false;
        }

        is_ready
    }
}

//
// SinkInfo
//

impl SinkInfos {
    fn init(&mut self) {
        for info in &mut self.0 {
            info.init();
        }
    }
    
    fn is_available(&mut self, index: usize) -> bool {
        self.0[index].is_available()
    }

    fn is_any_sink_available(&mut self) -> bool {
        for sink in &self.0 {
            if sink.is_available() {
                return true;
            }
        }

        return false;
    }

    fn send(&mut self, index: usize) {
        self.0[index].send();
    }

    fn request_source(&mut self, index: usize, n_request: u64) -> bool {
        self.0[index].request_source(n_request)
    }
}

impl SinkInfo {
    pub(crate) fn new(dst_id: NodeId) -> Self {
        Self {
            _dst_id: dst_id,
            n_request: 0,
            n_sent: 0,
            update_ticks: 0,
        }
    }

    fn init(&mut self) {
        self.n_request = 0;
        self.n_sent = 0;
    }

    fn is_available(&self) -> bool {
        self.n_sent < self.n_request
    }

    fn send(&mut self) {
        self.n_sent += 1;
    }

    fn request_source(&mut self, n_request: u64) -> bool {
        if self.n_request < n_request {
            self.n_request = n_request;
            true
        } else {
            false
        }
    }
}

//
// Tasks builder
//

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
        graph: &mut Graph,
    ) -> TaskId<O>
    where
        I: FlowIn<I>,
        O: Clone + 'static
    {
        let id = graph.push_input::<O>();
        let task_item = TaskNodeBuilder::new(id.clone(), task, src_nodes, graph, self);
    
        assert_eq!(id.index(), self.tasks.len());

        self.tasks.push(Box::new(task_item));

        id
    }

    pub(crate) fn add_sink<O: 'static>(
        &mut self,
        src_id: TaskId<O>,
        dst_id: NodeId,
        dst_index: usize,
    ) -> Box<dyn SourceTrait<O>> {
        let task = &mut self.tasks[src_id.index()];

        unsafe { 
            task.add_sink(dst_id, dst_index).unwrap::<Box<dyn SourceTrait<O>>>()
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
        graph: &mut Graph,
        tasks: &mut TasksBuilder,
    ) -> Self {
        let mut source_infos = Vec::new();
        let source = I::new_source(
            id.id(), 
            src_nodes, 
            &mut source_infos, 
            graph, 
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
    O: Clone + 'static
{
    unsafe fn add_sink(&mut self, dst_id: NodeId, dst_index: usize) -> UnsafePtr {
        let src_index = self.sinks.len();

        let (source, sink) = 
            task_channel(self.id.clone(), src_index, dst_id, dst_index);

        self.sinks.push(sink);
        self.sink_info.push(SinkInfo::new(dst_id));

        UnsafePtr::wrap(source)
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
    fn call(&mut self, source: &mut ()) -> Result<Out<()>> {
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
    F: FnMut(&mut I::Source) -> Result<Out<O>> + Send + 'static
{
    fn call(&mut self, source: &mut I::Source) -> Result<Out<O>> {
        self(source)
    }
}
