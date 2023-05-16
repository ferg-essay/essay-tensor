use std::{sync::Mutex, mem};

use super::{data::{FlowIn, GraphData}, flow::{TaskIdBare, TaskId}, dispatch::{Dispatcher}};

#[derive(Debug)]
pub struct TaskErr;

pub type Result<T> = std::result::Result<T, TaskErr>;

pub enum Out<T> {
    None,
    Some(T),
    Pending,
}

pub trait Source<T> : Sized {
    fn next(&mut self) -> Out<T>;

    fn push(&mut self, value: T);
}

pub struct SourceImpl<T> {
    item: Out<T>,
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
    fn add_output_arrow(&mut self, id: TaskIdBare);

    fn new_data(&self, data: &mut GraphData);

    fn init(
        &mut self, 
        data: &mut GraphData, 
        dispatcher: &mut Dispatcher,
    );

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
        id: TaskId<Out>,
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
    fn add_output_arrow(&mut self, id: TaskIdBare) {
        self.arrows_out.push(id);
    }

    fn new_data(&self, data: &mut GraphData) {
        data.push::<O>(self.arrows_out.len())
    }

    fn init(
        &mut self, 
        data: &mut GraphData, 
        dispatcher: &mut Dispatcher,
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
        dispatcher: &mut Dispatcher
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

impl<In> FlowNode for InputTask<In>
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
}

impl<T> Out<T> {
    #[inline]
    pub fn is_none(&self) -> bool {
        match self {
            Out::None => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_pending(&self) -> bool {
        match self {
            Out::Pending => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_some(&self) -> bool {
        match self {
            Out::Some(_) => true,
            _ => false,
        }
    }

    #[inline]
    pub fn take(&mut self) -> Self {
        mem::replace(self, Out::Pending)
    }

    #[inline]
    pub fn replace(&mut self, value: Self) -> Self {
        mem::replace(self, value)
    }
}

impl<T> Default for Out<T> {
    fn default() -> Self {
        Out::None
    }
}

impl<T> Default for SourceImpl<T> {
    fn default() -> Self {
        Self { 
            item: Default::default() 
        }
    }
}
impl<T> Source<T> for SourceImpl<T> {
    fn next(&mut self) -> Out<T> {
        self.item.take()        
    }

    fn push(&mut self, value: T) {
        assert!(self.item.is_none());

        self.item.replace(Out::Some(value));
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
