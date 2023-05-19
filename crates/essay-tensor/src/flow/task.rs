use std::{sync::{Mutex, MutexGuard, Arc, mpsc::Sender}, any::TypeId, ops::Deref, cell::UnsafeCell, marker::PhantomData};

use super::{
    data::{FlowIn},
    dispatch::{Dispatcher}, graph::{NodeId, TaskId, Graph}, 
    source::{Out, SinkTrait, SourceTrait, task_channel, SourceErr, Source}, 
    ptr::Ptr, flow::OutputData
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
    id: NodeId,

    source_info: Arc<Mutex<SourceInfos>>,
    sink_info: Arc<Mutex<SinkInfos>>,

    outer: Mutex<Box<dyn TaskOuter>>,
    inner: Mutex<Box<dyn TaskInner>>,
}

trait TaskPtr {}

trait TaskOuter {
    fn init(&mut self);

    fn request(
        &mut self, 
        src_index: usize,
        n_request: u64,
        waker: &mut Dispatcher,
    );

    fn complete(
        &mut self, 
        waker: &mut Dispatcher,
    );

    fn wake(
        &mut self, 
    ) -> Out<()>;

    fn update(
        &mut self, 
        // data: &mut GraphData, 
        dispatcher: &mut Dispatcher
    );
}

trait TaskInner {
    // unsafe fn source(&mut self, dst: NodeId, type_id: TypeId) -> Ptr;

    fn init(&mut self);

    fn wake(
        &mut self, 
        dispatcher: &mut Dispatcher,
    ) -> Out<()>;

    fn execute(&mut self, waker: &mut Dispatcher) -> Result<Out<()>>;
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
    sent: u64,

    update_ticks: u64,
}

struct SinkInfos(Vec<SinkInfo>);

type BoxTask<In, Out> = Box<dyn Task<In, Out>>;

struct TaskOuterNode<O> {
    id: TaskId<O>,

    state: NodeState,

    // in_nodes: I::Nodes,
    // arrows_out: Vec<NodeId>,

    // inner: Mutex<TaskInnerNode<In, Out>>,

    source_info: Arc<Mutex<SourceInfos>>,
    sink_info: Arc<Mutex<SinkInfos>>,
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

#[derive(Copy, Clone, Debug, PartialEq)]
enum NodeState {
    Idle,

    Active, // currently dispatched

    WaitingIn, // waiting for input
    WaitingOut, // waiting for output to clear (backpressure)
    WaitingInOut, // waiting for both input and output

    Complete,
}

//
// Task builder structs
//

pub struct TasksBuilder {
    tasks: Vec<Box<dyn TaskNodeBuilderTrait>>,
}

pub(crate) trait TaskNodeBuilderTrait {
    unsafe fn add_sink(&mut self, dst_id: NodeId, dst_index: usize) -> Ptr;

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

    pub(crate) fn wake(
        &self, 
        id: NodeId,
        // graph: &mut TaskGraph,
        dispatcher: &mut Dispatcher,
    ) {
        todo!();
        self.tasks[id.index()].wake(dispatcher);
    }

    pub(crate) fn request(
        &self, 
        id: NodeId,
        src_index: usize,
        n_request: u64,
        dispatcher: &mut Dispatcher,
    ) {
        self.tasks[id.index()].request(src_index, n_request, dispatcher)
    }

    pub(crate) fn complete(
        &self, 
        id: NodeId,
        waker: &mut Dispatcher,
    ) {
        self.tasks[id.index()].complete(waker);
    }

    pub(crate) fn update(
        &mut self, 
        id: NodeId,
        dispatcher: &mut Dispatcher, 
    ) {
        self.tasks[id.index()].update(dispatcher)
    }

    pub(crate) fn execute(
        &mut self, 
        id: NodeId,
        dispatcher: &mut Dispatcher, 
    ) {
        self.tasks[id.index()].execute(dispatcher)
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
            id: builder.id.id(),
            source_info,
            sink_info,
            outer: Mutex::new(Box::new(outer)),
            inner: Mutex::new(Box::new(inner)),
        }
    }

    fn init(&self) {
        self.outer.lock().unwrap().init();
        self.inner.lock().unwrap().init();
    }

    pub(crate) fn request(
        &self, 
        src_index: usize,
        n_request: u64,
        waker: &mut Dispatcher,
    ) {
        self.outer.lock().unwrap().request(src_index, n_request, waker);
    }

    pub(crate) fn complete(
        &self, 
        waker: &mut Dispatcher,
    ) {
        self.outer.lock().unwrap().complete(waker);
    }

    fn wake(
        &self, 
        dispatcher: &mut Dispatcher,
    ) {
        self.outer.lock().unwrap().wake();
    }

    fn update(
        &mut self, 
        waker: &mut Dispatcher, 
    ) {
        self.outer.lock().unwrap().update(waker);
    }

    fn execute(
        &mut self, 
        dispatcher: &mut Dispatcher, 
    ) {
        match self.inner.lock().unwrap().execute(dispatcher) {
            Ok(_) => {
            },
            Err(_) => todo!(),
        }
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

    fn execute(&mut self, dispatcher: &mut Dispatcher) -> Result<()> {
        todo!();
        /*
        match self.inner.lock().unwrap().execute()? {
            Out::Some(out) => {
                    // data.write(&self.id, out);
    
                    // TODO: allow multi-buffer
                self.state = NodeState::WaitingOut;
                todo!();
    
                    for node in &self.arrows_out {
                        dispatcher.complete(*node, data);
                    }
            }
            Out::None => {
                self.state = NodeState::Complete;
            }
            Out::Pending => {
                todo!()
            }
        }
        */
    
        Ok(())
    
    }
}

impl<O> TaskOuter for TaskOuterNode<O>
where
    O: Clone + 'static
{

    /*
    fn new_data(&self, data: &mut GraphData) {
        data.push::<O>(self.arrows_out.len())
    }
    */

    fn init(
        &mut self, 
    ) {
        self.state = NodeState::Idle;

        // self.inner.lock().unwrap().init();
    }

    fn request(
        &mut self, 
        sink_index: usize,
        n_request: u64,
        waker: &mut Dispatcher,
    ) {
        if ! self.sink_info.lock().unwrap().request(sink_index, n_request) {
            return;
        }

        match self.state {
            NodeState::Idle => {
                self.state = NodeState::WaitingIn;

                if self.source_info.lock().unwrap().request(waker) {
                    self.state = NodeState::Active;
                    waker.spawn(self.id.id());
                }
                /*
                if self.inner.lock().unwrap().fill_input(&self.in_nodes) {
                    self.state = NodeState::Active;
                    dispatcher.spawn(self.id.id());

                    Out::Some(())
                } else {
                    Out::Pending
                }
                */
            },
            NodeState::Active => {
            },
            NodeState::WaitingIn => {
            },
            NodeState::WaitingOut => {
            },
            NodeState::WaitingInOut => {
            },
            NodeState::Complete => {
            },
        }
    }

    fn wake(
        &mut self, 
    ) -> Out<()> {
        match self.state {
            NodeState::Idle => {
                self.state = NodeState::WaitingIn;
                /*
                if self.inner.lock().unwrap().fill_input(&self.in_nodes) {
                    self.state = NodeState::Active;
                    dispatcher.spawn(self.id.id());

                    Out::Some(())
                } else {
                    Out::Pending
                }
                */
                Out::Pending
            },
            NodeState::Active => {
                Out::Some(())
            },
            NodeState::WaitingIn => {
                Out::Pending
            },
            NodeState::WaitingOut => Out::Some(()),
            NodeState::WaitingInOut => Out::Some(()),
            NodeState::Complete => Out::None,
        }
    }

    fn update(
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
                if self.source_info.lock().unwrap().request(waker) {
                    self.state = NodeState::Active;
                    waker.spawn(self.id.id());
                }
            },
            NodeState::WaitingOut => todo!(),
            NodeState::WaitingInOut => todo!(),
            NodeState::Complete => todo!(),
        }
    }

    fn complete(
        &mut self, 
        waker: &mut Dispatcher,
    ) {
        match self.state {
            NodeState::Active => {
                self.state = NodeState::WaitingOut;
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
        self.sink_info.lock().unwrap().is_sink_available(next_index)
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
        waker.add_update(self.sinks[index].id());
    }
} 

impl<I, O> TaskInner for TaskInnerNode<I, O>
where
    I: FlowIn<I>,
    O: Clone + 'static
{
    fn init(&mut self) {
        self.task.init();
    }

    fn wake(&mut self, waker: &mut Dispatcher) -> Out<()> {
        if I::fill_source(&mut self.source, waker) {
            waker.spawn(self.id.id());

            Out::Some(())
        } else {
            println!("Wake Pending {:?}", self.id);
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
    fn request(&mut self, waker: &mut Dispatcher) -> bool {
        let mut is_ready = true;

        for info in &mut self.0 {
            if ! info.request(waker) {
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

    fn request(&mut self, waker: &mut Dispatcher) -> bool {
        let mut is_ready = true;

        let n_request = self.n_read + self.request_window as u64;

        if self.n_request < n_request {
            self.n_request = n_request;
            waker.request(self.src_id, self.src_index, n_request);
            is_ready = false;
        }

        is_ready
    }
}

//
// SinkInfo
//

impl SinkInfos {
    fn is_sink_available(&mut self, index: usize) -> bool {
        self.0[index].is_available()
    }

    fn send(&mut self, index: usize) {
        self.0[index].send();
    }

    fn request(&mut self, index: usize, n_request: u64) -> bool {
        self.0[index].request(n_request)
    }
}

impl SinkInfo {
    pub(crate) fn new(dst_id: NodeId) -> Self {
        Self {
            _dst_id: dst_id,
            n_request: 0,
            sent: 0,
            update_ticks: 0,
        }
    }

    fn is_available(&self) -> bool {
        self.sent < self.n_request
    }

    fn send(&mut self) {
        self.sent += 1;
    }

    fn request(&mut self, n_request: u64) -> bool {
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

        // Box::new(TaskInnerNode(id, task),
        // Box::new(TaskOuterNode(id, task)),
    
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
    unsafe fn add_sink(&mut self, dst_id: NodeId, dst_index: usize) -> Ptr {
        // self.arrows_out.push(dst_id);
        let src_index = self.sinks.len();

        let (source, sink) = 
            task_channel(self.id.clone(), src_index, dst_id, dst_index);

        self.sinks.push(sink);
        self.sink_info.push(SinkInfo::new(dst_id));

        Ptr::wrap(source)
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
