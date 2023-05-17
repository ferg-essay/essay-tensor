use std::{sync::Mutex};

use super::{
    data::{FlowIn, GraphData, Out},
    dispatch::{Dispatcher}, graph::{TaskIdBare, TaskId}
};

#[derive(Debug)]
pub struct TaskErr;

pub type Result<T> = std::result::Result<T, TaskErr>;

pub struct Source<T> {
    item: Out<T>,
}

pub trait Task<I, O> : Send + 'static
where
    I: FlowIn<I> + 'static,
    O: 'static,
{
    fn init(&mut self) {}
    
    fn execute(&mut self, source: &mut I::Source) -> Result<Out<O>>;
}

pub trait TaskOuter {
    fn add_output_arrow(&mut self, id: TaskIdBare);

    fn new_data(&self, data: &mut GraphData);

    fn init(
        &mut self, 
        data: &mut GraphData, 
        dispatcher: &mut Dispatcher,
    );

    fn wake(
        &mut self, 
        // graph: &mut TaskGraph,
        dispatcher: &mut Dispatcher,
        data: &mut GraphData,
    ) -> Out<()>;

    fn update(
        &mut self, 
        data: &mut GraphData, 
        dispatcher: &mut Dispatcher
    );

    fn complete(&mut self, dispatcher: &Dispatcher) -> bool;

    fn execute(&mut self, data: &mut GraphData, waker: &mut Dispatcher) -> Result<()>;
}

type BoxTask<In, Out> = Box<dyn Task<In, Out>>;

pub struct TaskNode<In: FlowIn<In>, Out> {
    id: TaskId<Out>,

    state: NodeState,

    arrows_in: In::Nodes,
    arrows_out: Vec<TaskIdBare>,

    inner: Mutex<TaskInnerNode<In, Out>>,
}

struct TaskInnerNode<I, Out>
where
    I: FlowIn<I>
{
    task: BoxTask<I, Out>,
    source: I::Source,
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

impl<I, O> TaskNode<I, O>
where
    I: FlowIn<I> + 'static, // ArrowData<Key, Value=In> + 'static,
    O: 'static
{
    pub fn new(
        id: TaskId<O>,
        task: impl Task<I, O>,
        input: I::Nodes, // BoxArrow<In>,
    ) -> Self {
        let source = I::new_source(&input);
        let inner : TaskInnerNode<I, O> = TaskInnerNode::new(
            Box::new(task),
            source
        );

        Self {
            id: id,
            state: NodeState::WaitingIn,
            arrows_in: input,
            arrows_out: Default::default(),
            inner: Mutex::new(inner),
        }
    }
}

impl<I, O> TaskOuter for TaskNode<I, O>
where
    I: FlowIn<I> + 'static, // ArrowData<Value=In>,
    O: Clone + 'static
{
    fn add_output_arrow(&mut self, id: TaskIdBare) {
        self.arrows_out.push(id);
    }

    fn new_data(&self, data: &mut GraphData) {
        data.push::<O>(self.arrows_out.len())
    }

    fn init(
        &mut self, 
        _data: &mut GraphData, 
        _dispatcher: &mut Dispatcher,
    ) {
        self.state = NodeState::Idle;

        self.inner.lock().unwrap().init();

        /*
        if self.inner.lock().unwrap().fill_input(&self.arrows_in, data) {
            self.state = NodeState::Active;
            dispatcher.spawn(self.id.id());
        }
        */
    }

    fn wake(
        &mut self, 
        // graph: &mut TaskGraph, 
        dispatcher: &mut Dispatcher,
        data: &mut GraphData,
    ) -> Out<()> {
        match self.state {
            NodeState::Idle => {
                self.state = NodeState::WaitingIn;

                if self.inner.lock().unwrap().fill_input(&self.arrows_in, data) {
                    self.state = NodeState::Active;
                    dispatcher.spawn(self.id.id());

                    Out::Some(())
                } else {
                    Out::Pending // I::wake(&self.arrows_in, graph, dispatcher, data);
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
        data: &mut GraphData, 
        dispatcher: &mut Dispatcher
    ) {
        match self.state {
            NodeState::Idle => {},
            NodeState::Active => {},
            NodeState::WaitingIn => {
                if self.inner.lock().unwrap().fill_input(&self.arrows_in, data) {
                    self.state = NodeState::Active;
                    dispatcher.spawn(self.id.id());
                }
            },
            NodeState::WaitingOut => {},
            NodeState::WaitingInOut => todo!(),
            NodeState::Complete => {},
        }
    }

    fn complete(&mut self, _dispatcher: &Dispatcher) -> bool {
        todo!()
    }

    fn execute(&mut self, data: &mut GraphData, dispatcher: &mut Dispatcher) -> Result<()> {
        match self.inner.lock().unwrap().execute()? {
            Out::Some(out) => {
                // self.output.push_back(out);
                data.write(&self.id, out);

                // TODO: allow multi-buffer
                self.state = NodeState::WaitingOut;

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

        Ok(())
    }
}

impl<I, O> TaskInnerNode<I, O>
where
    I: FlowIn<I>,
    O: 'static
{
    fn new(
        task: BoxTask<I, O>,
        source: I::Source,
    ) -> Self {
        Self {
            task: task,
            source,
        }
    }

    fn init(&mut self) {
        // task.init()
    }

    fn fill_input(&mut self, arrows_in: &I::Nodes, data: &mut GraphData) -> bool {
        I::fill_input(&mut self.source, arrows_in, data)
    }

    fn execute(&mut self) -> Result<Out<O>> {
        self.task.execute(&mut self.source)
    }
} 

pub struct InputTask<In: FlowIn<In>> {
    _id: In::Nodes,

    arrows_out: Vec<TaskIdBare>,
}

impl<In: FlowIn<In>> InputTask<In> {
    pub fn new(id: In::Nodes) -> Self {
        Self {
            _id: id,
            arrows_out: Vec::new(),
        }
    }
}

impl<In> TaskOuter for InputTask<In>
where
    In: FlowIn<In> + 'static,
{
    fn add_output_arrow(&mut self, id: TaskIdBare) {
        self.arrows_out.push(id);
    }

    fn new_data(&self, data: &mut GraphData) {
        data.push::<In>(self.arrows_out.len())
    }

    fn init(
        &mut self, 
        data: &mut GraphData, 
        waker: &mut Dispatcher,
    ) {
        for node in &self.arrows_out {
            waker.complete(*node, data);
        }
    }

    fn update(
        &mut self, 
        _data: &mut GraphData, 
        _dispatcher: &mut Dispatcher
    ) {
    }

    fn complete(&mut self, _dispatcher: &Dispatcher) -> bool {
        todo!()
    }

    fn execute(&mut self, _data: &mut GraphData, _waker: &mut Dispatcher) -> Result<()> {
        Ok(())
    }

    fn wake(
        &mut self, 
        // graph: &mut TaskGraph,
        _dispatcher: &mut Dispatcher,
        _data: &mut GraphData,
    ) -> Out<()> {
        Out::Some(())
    }
}

pub struct NilTask {
}

impl NilTask {
    pub fn new() -> Self {
        Self {
        }
    }
}

impl TaskOuter for NilTask {
    fn add_output_arrow(&mut self, _id: TaskIdBare) {
    }

    fn new_data(&self, data: &mut GraphData) {
        data.push::<()>(0);
    }

    fn init(
        &mut self, 
        _data: &mut GraphData, 
        _waker: &mut Dispatcher,
    ) {
    }

    fn update(
        &mut self, 
        _data: &mut GraphData, 
        _dispatcher: &mut Dispatcher
    ) {
    }

    fn complete(&mut self, _dispatcher: &Dispatcher) -> bool {
        todo!()
    }

    fn execute(&mut self, _data: &mut GraphData, _waker: &mut Dispatcher) -> Result<()> {
        Ok(())
    }

    fn wake(
        &mut self, 
        // graph: &mut TaskGraph,
        _dispatcher: &mut Dispatcher,
        _data: &mut GraphData,
    ) -> Out<()> {
        Out::Some(())
    }
}

impl<T> Default for Source<T> {
    fn default() -> Self {
        Self { 
            item: Default::default() 
        }
    }
}
impl<T> Source<T> {
    pub(crate) fn new() -> Self {
        Self {
            item: Out::Pending,
        }
    }

    pub fn next(&mut self) -> Out<T> {
        self.item.take()        
    }

    pub fn push(&mut self, value: T) {
        assert!(self.item.is_none());

        self.item.replace(Out::Some(value));
    }

    #[inline]
    pub(crate) fn is_some(&self) -> bool {
        self.item.is_some()
    }

    #[inline]
    pub(crate) fn is_none(&self) -> bool {
        self.item.is_none()
    }
}

impl<I, O, F> Task<I, O> for F
where
    I: FlowIn<I> + 'static,
    O: 'static,
    F: FnMut(&mut I::Source) -> Result<Out<O>> + Send + 'static
{
    fn execute(&mut self, source: &mut I::Source) -> Result<Out<O>> {
        self(source)
    }
}
