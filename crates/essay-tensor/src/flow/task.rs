use std::{sync::{Mutex, MutexGuard, Arc}, any::TypeId, ops::Deref, cell::UnsafeCell};

use super::{
    data::{FlowIn},
    dispatch::{Dispatcher}, graph::{NodeId, TaskId, Graph}, 
    source::{Out, SinkTrait, SourceTrait, task_channel, SourceErr, Source}, 
    ptr::Ptr
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
    source_info: Arc<Mutex<Vec<SourceInfo>>>,
    sink_info: Arc<Mutex<Vec<SinkInfo>>>,

    outer: Mutex<Box<dyn TaskOuter>>,
    inner: Mutex<Box<dyn TaskInner>>,
}

trait TaskPtr {}

trait TaskOuter {
    fn init(&mut self);

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

struct SourceInfo {
    request_window: u32,

    read: u64,
    requested: u64,
}

struct SinkInfo {
    requested: Vec<u64>,

    update_ticks: u64,
}

type BoxTask<In, Out> = Box<dyn Task<In, Out>>;

struct TaskOuterNode<O> {
    id: TaskId<O>,

    state: NodeState,

    // in_nodes: I::Nodes,
    // arrows_out: Vec<NodeId>,

    // inner: Mutex<TaskInnerNode<In, Out>>,

    source_info: Arc<Mutex<Vec<SourceInfo>>>,
    sink_info: Arc<Mutex<Vec<SinkInfo>>>,
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

    source_info: Arc<Mutex<Vec<SourceInfo>>>,
    sink_info: Arc<Mutex<Vec<SinkInfo>>>,
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
    unsafe fn add_sink(&mut self, dst_id: NodeId) -> Ptr;

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
    ) -> Out<()> {
        self.tasks[id.index()].wake(dispatcher)
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
        let source_info = builder.source_info.drain(..).collect();
        let source_info = Arc::new(Mutex::new(source_info));

        let sink_info = builder.sink_info.drain(..).collect();
        let sink_info = Arc::new(Mutex::new(sink_info));

        let outer = TaskOuterNode::new(builder, &source_info, &sink_info);
        let inner = TaskInnerNode::new(builder, &source_info, &sink_info);

        Self {
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

    fn wake(
        &self, 
        dispatcher: &mut Dispatcher,
    ) -> Out<()> {
        match self.outer.lock().unwrap().wake() {
            Out::None => Out::None,
            Out::Some(_) => Out::Some(()), // task is active
            Out::Pending => {
                // since task is inactive, can lock inner
                self.inner.lock().unwrap().wake(dispatcher)
            }
        }
    }

    fn update(
        &mut self, 
        dispatcher: &mut Dispatcher, 
    ) {
        todo!();
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
        source_info: &Arc<Mutex<Vec<SourceInfo>>>,
        sink_info: &Arc<Mutex<Vec<SinkInfo>>>,
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
    /*
        id: TaskId<O>,
        task: impl Task<I, O>,
        arrows_in: I::Nodes, // BoxArrow<In>,
        graph: &mut Graph,
        tasks: &mut Tasks,
    ) -> Self {
        */

        /*
        let source = I::add_source(
            id.id(), 
            &arrows_in,
            graph,
            tasks
        );

        let inner : TaskInnerNode<I, O> = TaskInnerNode::new(
            id,
            Box::new(task),
            source
        );
        */

        /*
        //let outer_source = I::new_source(&arrows_in);
        Self {
            id: id,
            state: NodeState::WaitingIn,
            arrows_in,
            arrows_out: Default::default(),
            inner: Mutex::new(inner),

            // source: outer_source,
        }
        */
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
        dispatcher: &mut Dispatcher
    ) {
        match self.state {
            NodeState::Idle => {},
            NodeState::Active => {},
            NodeState::WaitingIn => {
                todo!();
                /*
                if self.inner.lock().unwrap().fill_input(&self.arrows_in, data) {
                    self.state = NodeState::Active;
                    dispatcher.spawn(self.id.id());
                }
                */
            },
            NodeState::WaitingOut => {},
            NodeState::WaitingInOut => todo!(),
            NodeState::Complete => {},
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
        source_info: &Arc<Mutex<Vec<SourceInfo>>>,
        sink_info: &Arc<Mutex<Vec<SinkInfo>>>,
    ) -> Self {
        Self {
            id: builder.id.clone(), 
            task: builder.task.take().unwrap(),
            source: builder.source.take().unwrap(),
            sinks: Default::default(),

            sink_index: 0,
            source_info: source_info.clone(),
            sink_info: sink_info.clone(),
        }
    }

    fn send(&mut self, value: Option<O>) {
        println!("Send {:?}", value.is_some());
    }

    fn send_all(&mut self, value: Option<O>) {
        println!("Send-All {:?}", value.is_some());
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

    fn wake(&mut self, dispatcher: &mut Dispatcher) -> Out<()> {
        dispatcher.spawn(self.id.id());

        Out::Some(())
    }

    fn execute(&mut self, waker: &mut Dispatcher) -> Result<Out<()>> {
        while I::fill_source(&mut self.source) {
            match self.task.call(&mut self.source) {
                Ok(Out::Some(value)) => self.send(Some(value)),
                Ok(Out::None) => {
                    self.send_all(None);
                    return Ok(Out::None);
                },
                Ok(Out::Pending) => {
                    return Ok(Out::Pending);
                },
                Err(err) => {
                    panic!("Error from task");
                    // return Ok(Out::Pending); // Err(SourceErr::Pending)
                },
            }
        }

        Ok(Out::Pending)
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
        dst_id: NodeId
    ) -> Box<dyn SourceTrait<O>> {
        let task = &mut self.tasks[src_id.index()];

        unsafe { 
            task.add_sink(dst_id).unwrap::<Box<dyn SourceTrait<O>>>()
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
        let source = I::add_source(id.id(), src_nodes, graph, tasks);

        Self {
            id: id.clone(),
            task: Some(task),

            source: Some(source),
            source_info: Vec::default(),

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
    unsafe fn add_sink(&mut self, dst_id: NodeId) -> Ptr {
        // self.arrows_out.push(dst_id);

        let (source, sink) = 
            task_channel(self.id.clone(), dst_id);

        self.sinks.push(sink);

        Ptr::wrap(source)
    }

    fn build(&mut self) -> TaskNode {
        TaskNode::new(self)
    }
}

//
// task implementations
//

pub struct NilTask;

impl Task<(), ()> for NilTask {
    fn call(&mut self, source: &mut ()) -> Result<Out<()>> {
        todo!()
    }
}

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
