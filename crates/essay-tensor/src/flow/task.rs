use std::{sync::Mutex};

use super::{data::{FlowIn, GraphData}, flow::{TaskId, TypedTaskId}, dispatch::{Dispatcher, Waker}};

pub struct TaskErr;
pub type Result<T> = std::result::Result<T, TaskErr>;

pub enum Out<T> {
    None,
    Some(T),
    Pending,
}

pub trait Task<I, O> : Send + 'static
where
    I: 'static,
    O: 'static,
{
    fn init(&mut self) {}
    
    fn execute(&mut self, input: I) -> Result<Out<O>>;
}

pub trait FlowNode {
    fn add_output_arrow(&mut self, id: TaskId);

    fn new_data(&self, data: &mut GraphData);

    fn init(
        &mut self, 
        data: &mut GraphData, 
        dispatcher: &mut dyn Dispatcher,
        waker: &mut Waker,
    );

    fn update(
        &mut self, 
        data: &mut GraphData, 
        dispatcher: &mut dyn Dispatcher
    );

    fn complete(&mut self, dispatcher: &dyn Dispatcher) -> bool;

    fn execute(&mut self, data: &mut GraphData, waker: &mut Waker) -> Result<()>;
}

trait NodeInner {
    fn execute(&mut self, waker: &Waker);
}

type BoxTask<In, Out> = Box<dyn Task<In, Out>>;

pub struct TaskNode<In: FlowIn<In>, Out> {
    id: TypedTaskId<Out>,

    state: NodeState,

    arrows_in: In::Nodes,
    arrows_out: Vec<TaskId>,

    inner: Mutex<TaskInner<In, Out>>,
}

struct TaskInner<In, Out> {
    task: BoxTask<In, Out>,
    input: Option<In>,
}

enum NodeState {
    Active, // currently dispatched

    WaitingIn, // waiting for input
    WaitingOut, // waiting for output to clear (backpressure)
    WaitingInOut, // waiting for both input and output

    Complete,
}

impl<In, Out> TaskNode<In, Out>
where
    In: FlowIn<In> + 'static, // ArrowData<Key, Value=In> + 'static,
    Out: 'static
{
    pub fn new(
        id: TypedTaskId<Out>,
        task: impl Task<In, Out>,
        input: In::Nodes, // BoxArrow<In>,
    ) -> Self {
        let inner : TaskInner<In, Out> = TaskInner::new(Box::new(task));

        Self {
            id: id,
            state: NodeState::WaitingIn,
            arrows_in: input,
            arrows_out: Default::default(),
            inner: Mutex::new(inner),
        }
    }
}

impl<I, O> FlowNode for TaskNode<I, O>
where
    I: FlowIn<I> + 'static, // ArrowData<Value=In>,
    O: Clone + 'static
{
    fn add_output_arrow(&mut self, id: TaskId) {
        self.arrows_out.push(id);
    }

    fn new_data(&self, data: &mut GraphData) {
        data.push::<O>(self.arrows_out.len())
    }

    fn init(
        &mut self, 
        data: &mut GraphData, 
        dispatcher: &mut dyn Dispatcher,
        _waker: &mut Waker,
    ) {
        self.state = NodeState::WaitingIn;

        if I::is_available(&self.arrows_in, data) {
            let input = I::read(&self.arrows_in, data);
            self.state = NodeState::Active;
            self.inner.lock().unwrap().input.replace(input);

            dispatcher.spawn(self.id.id());
        }
    }

    fn update(
        &mut self, 
        data: &mut GraphData, 
        dispatcher: &mut dyn Dispatcher
    ) {
        match self.state {
            NodeState::Active => {},
            NodeState::WaitingIn => {
                if I::is_available(&self.arrows_in, data) {
                    let input = I::read(&self.arrows_in, data);
                    self.state = NodeState::Active;
                    self.inner.lock().unwrap().input.replace(input);
        
                    dispatcher.spawn(self.id.id());
                }
            },
            NodeState::WaitingOut => {},
            NodeState::WaitingInOut => todo!(),
            NodeState::Complete => {},
        }
    }

    fn complete(&mut self, _dispatcher: &dyn Dispatcher) -> bool {
        todo!()
    }

    fn execute(&mut self, data: &mut GraphData, waker: &mut Waker) -> Result<()> {
        match self.inner.lock().unwrap().execute()? {
            Out::Some(out) => {
                // self.output.push_back(out);
                data.write(&self.id, out);

                // TODO: allow multi-buffer
                self.state = NodeState::WaitingOut;

                for node in &self.arrows_out {
                    waker.complete(*node, data);
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

impl<I, O> TaskInner<I, O>
where
    I: 'static,
    O: 'static
{
    fn new(task: BoxTask<I, O>) -> Self {
        Self {
            task: task,
            input: None,
        }
    }

    fn execute(&mut self) -> Result<Out<O>> {
        let input = self.input.take().unwrap();

        self.task.execute(input)
    }
} 

pub struct InputNode<In: FlowIn<In>> {
    _id: In::Nodes,

    arrows_out: Vec<TaskId>,
}

impl<In: FlowIn<In>> InputNode<In> {
    pub fn new(id: In::Nodes) -> Self {
        Self {
            _id: id,
            arrows_out: Vec::new(),
        }
    }
}

impl<In> FlowNode for InputNode<In>
where
    In: FlowIn<In> + 'static,
{
    fn add_output_arrow(&mut self, id: TaskId) {
        self.arrows_out.push(id);
    }

    fn new_data(&self, data: &mut GraphData) {
        data.push::<In>(self.arrows_out.len())
    }

    fn init(
        &mut self, 
        data: &mut GraphData, 
        _dispatcher: &mut dyn Dispatcher,
        waker: &mut Waker,
    ) {
        for node in &self.arrows_out {
            waker.complete(*node, data);
        }
    }

    fn update(
        &mut self, 
        _data: &mut GraphData, 
        _dispatcher: &mut dyn Dispatcher
    ) {
    }

    fn complete(&mut self, _dispatcher: &dyn Dispatcher) -> bool {
        todo!()
    }

    fn execute(&mut self, _data: &mut GraphData, _waker: &mut Waker) -> Result<()> {
        Ok(())
    }
}

impl<I, O, F> Task<I, O> for F
where
    I: 'static,
    O: 'static,
    F: FnMut(I) -> Option<O> + Send + 'static
{
    fn execute(&mut self, input: I) -> Result<Out<O>> {
        match self(input) {
            Some(out) => Ok(Out::Some(out)),
            None => Ok(Out::None),
        }
    }
}
