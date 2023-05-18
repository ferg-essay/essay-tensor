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

struct TaskNode {
    source_info: Arc<Mutex<Vec<SourceInfo>>>,
    sink_info: Arc<Mutex<SinkInfo>>,

    outer: Mutex<Box<dyn TaskOuter>>,
    inner: Mutex<Box<dyn TaskInner>>,
}

trait TaskPtr {}

trait TaskOuter {
    fn add_output_arrow(&mut self, dst_id: NodeId);

    // fn new_data(&self, data: &mut GraphData);

    fn init(&mut self);

    fn wake(
        &mut self, 
        // graph: &mut TaskGraph,
        dispatcher: &mut Dispatcher,
        // data: &mut GraphData,
    ) -> Out<()>;

    fn update(
        &mut self, 
        // data: &mut GraphData, 
        dispatcher: &mut Dispatcher
    );

    fn complete(&mut self, dispatcher: &Dispatcher) -> bool;
}

trait TaskInner {
    // unsafe fn source(&mut self, dst: NodeId, type_id: TypeId) -> Ptr;

    fn init(&mut self);

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

struct TaskOuterNode<In: FlowIn<In>, Out> {
    id: TaskId<Out>,

    state: NodeState,

    arrows_in: In::Nodes,
    arrows_out: Vec<NodeId>,

    inner: Mutex<TaskInnerNode<In, Out>>,

    source_info: Arc<Mutex<Vec<SourceInfo>>>,
    sink_info: Arc<Mutex<SinkInfo>>,
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
    sink_info: Arc<Mutex<SinkInfo>>,
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

pub struct TasksBuilder {
    tasks: Vec<Box<dyn TaskNodeBuilderTrait>>,
}

pub(crate) trait TaskNodeBuilderTrait {
    unsafe fn add_sink(&mut self, dst_id: NodeId) -> Ptr;
}

pub(crate) struct TaskNodeBuilder<I, O>
where
    I: FlowIn<I>,
    O: 'static
{
    id: TaskId<O>,
    task: BoxTask<I, O>,

    source: I::Source,
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
    fn from_task<I: FlowIn<I>, O>(
        id: TaskId<O>,
        task: BoxTask<I, O>,
        source: I::Source,
    ) -> Self {
        todo!();
    }

    fn init(&self) {
        todo!();
    }

    fn wake(
        &self, 
        dispatcher: &mut Dispatcher,
    ) -> Out<()> {
        todo!();
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
        todo!();
    }
}

//
// TaskOuter
//

impl<I, O> TaskOuterNode<I, O>
where
    I: FlowIn<I> + 'static, // ArrowData<Key, Value=In> + 'static,
    O: 'static
{
    pub fn new(
        id: TaskId<O>,
        task: impl Task<I, O>,
        arrows_in: I::Nodes, // BoxArrow<In>,
        graph: &mut Graph,
        tasks: &mut Tasks,
    ) -> Self {
        todo!();

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
}

impl<I, O> TaskOuter for TaskOuterNode<I, O>
where
    I: FlowIn<I> + 'static, // ArrowData<Value=In>,
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

        self.inner.lock().unwrap().init();
    }

    fn wake(
        &mut self, 
        dispatcher: &mut Dispatcher,
        // data: &mut GraphData,
    ) -> Out<()> {
        match self.state {
            NodeState::Idle => {
                self.state = NodeState::WaitingIn;

                if self.inner.lock().unwrap().fill_input(&self.arrows_in) {
                    self.state = NodeState::Active;
                    dispatcher.spawn(self.id.id());

                    Out::Some(())
                } else {
                    Out::Pending
                }
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

    fn complete(&mut self, _dispatcher: &Dispatcher) -> bool {
        todo!()
    }

    fn add_output_arrow(&mut self, dst_id: NodeId) {
        todo!()
    }
}

impl<I, O> TaskOuterNode<I, O> 
where
    I: FlowIn<I>,
    O: 'static
{
    fn execute(&mut self, dispatcher: &mut Dispatcher) -> Result<()> {
        match self.inner.lock().unwrap().execute()? {
            Out::Some(out) => {
                todo!();
                // data.write(&self.id, out);

                // TODO: allow multi-buffer
                self.state = NodeState::WaitingOut;

                /*
                for node in &self.arrows_out {
                    dispatcher.complete(*node, data);
                }
                */
            }
            Out::None => {
                self.state = NodeState::Complete;
            }
            Out::Pending => {
                todo!()
            }
        }

        Ok(())
    }

}

//
// TaskInner
//

impl<I, O> TaskInnerNode<I, O>
where
    I: FlowIn<I>,
    O: 'static
{
    fn new(
        id: TaskId<O>,
        task: BoxTask<I, O>,
        source: I::Source,
    ) -> Self {
        todo!();
        /*
        Self {
            id, 
            task: task,
            source,
            sinks: Default::default(),
        }
        */
    }

    fn init(&mut self) {
        // task.init()
    }

    unsafe fn add_sink(&mut self, dst_id: NodeId) -> Ptr {
        let (source, sink) = task_channel(self.id.clone(), dst_id);

        self.sinks.push(sink);

        Ptr::wrap(source)
    }

    fn fill_input(&mut self, arrows_in: &I::Nodes) -> bool {
        todo!();
        // I::fill_input(&mut self.source, arrows_in, data)
    }

    fn execute(&mut self) -> Result<Out<O>> {
        while I::fill_source(&mut self.source) {
            match self.task.call(&mut self.source) {
                Ok(Out::Some(value)) => self.send(Some(value)),
                Ok(Out::None) => {
                    self.send_all(None);
                    return Ok(Out::None);
                },
                Ok(Out::Pending) => {
                    todo!();
                },
                Err(err) => {
                    panic!("Error from task");
                    // return Ok(Out::Pending); // Err(SourceErr::Pending)
                },
            }
        }

        Ok(Out::Pending)
    }

    fn send(&mut self, value: Option<O>) {
        println!("Send {:?}", value.is_some());
    }

    fn send_all(&mut self, value: Option<O>) {
        println!("Send-All {:?}", value.is_some());
    }
} 

//
// Tasks builder
//

impl TasksBuilder {
    pub(crate) fn new() -> Self {
        todo!()
    }

    pub(crate) fn push_task<I, O>(
        &mut self, 
        id: TaskId<O>, 
        task: BoxTask<I, O>,
        src_nodes: &I::Nodes,
        graph: &mut Graph,
    )
    where
        I: FlowIn<I>,
        O: 'static
    {
        let task_item = TaskNodeBuilder::new(id.clone(), task, src_nodes, graph, self);

        // Box::new(TaskInnerNode(id, task),
        // Box::new(TaskOuterNode(id, task)),
    
        assert_eq!(id.index(), self.tasks.len());

        self.tasks.push(Box::new(task_item));
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

    pub(crate) fn build(&self) -> Tasks {
        todo!()
    }
}

//
// TaskNode builder
//

impl<I, O> TaskNodeBuilder<I, O>
where
    I: FlowIn<I>,
    O: 'static
{
    fn new(
        id: TaskId<O>,
        task: Box<dyn Task<I, O>>,
        src_nodes: &I::Nodes,
        graph: &mut Graph,
        tasks: &mut TasksBuilder,
    ) -> Self {
        Self {
            id: id.clone(),
            task,

            source: I::add_source(id.id(), src_nodes, graph, tasks),
            source_info: Vec::default(),

            sinks: Vec::default(),
            sink_info: Vec::default(),
        }
    }
}

impl<I, O> TaskNodeBuilderTrait for TaskNodeBuilder<I, O>
where
    I: FlowIn<I>,
    O: 'static
{
    unsafe fn add_sink(&mut self, dst_id: NodeId) -> Ptr {
        // self.arrows_out.push(dst_id);

        let (source, sink) = 
            task_channel(self.id.clone(), dst_id);

        self.sinks.push(sink);

        Ptr::wrap(source)
    }
}

//
// task implementations
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
