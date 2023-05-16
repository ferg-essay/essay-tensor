use std::{sync::Mutex};

use super::{data::{FlowData, GraphData, Scalar}, flow::{TaskId, TypedTaskId}, dispatch::{Dispatcher, Waker}};

pub trait Task<In, Out> : Send + 'static
where
    In: 'static,
    Out: 'static,
{
    fn init(&mut self) {}
    
    fn execute(&mut self, input: In) -> Option<Out>;
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

    fn execute(&mut self, data: &mut GraphData, waker: &mut Waker);
}

trait NodeInner {
    fn execute(&mut self, waker: &Waker);
}

type BoxTask<In, Out> = Box<dyn Task<In, Out>>;

pub struct TaskNode<In: FlowData<In>, Out> {
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
    In: FlowData<In> + 'static, // ArrowData<Key, Value=In> + 'static,
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

impl<In, Out> FlowNode for TaskNode<In, Out>
where
    In: FlowData<In> + 'static, // ArrowData<Value=In>,
    Out: Clone + 'static
{
    fn add_output_arrow(&mut self, id: TaskId) {
        self.arrows_out.push(id);
    }

    fn new_data(&self, data: &mut GraphData) {
        data.push::<Out>(self.arrows_out.len())
    }

    fn init(
        &mut self, 
        data: &mut GraphData, 
        dispatcher: &mut dyn Dispatcher,
        _waker: &mut Waker,
    ) {
        self.state = NodeState::WaitingIn;

        if In::is_available(&self.arrows_in, data) {
            let input = In::read(&self.arrows_in, data);
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
                if In::is_available(&self.arrows_in, data) {
                    let input = In::read(&self.arrows_in, data);
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

    fn execute(&mut self, data: &mut GraphData, waker: &mut Waker) {
        match self.inner.lock().unwrap().execute() {
            Some(out) => {
                // self.output.push_back(out);
                data.write(&self.id, out);

                // TODO: allow multi-buffer
                self.state = NodeState::WaitingOut;

                for node in &self.arrows_out {
                    waker.complete(*node, data);
                }
            }
            None => {
                self.state = NodeState::Complete;
            }
        }
    }
}

impl<In, Out> TaskInner<In, Out>
where
    In: 'static,
    Out: 'static
{
    fn new(task: BoxTask<In, Out>) -> Self {
        Self {
            task: task,
            input: None,
        }
    }

    fn execute(&mut self) -> Option<Out> {
        let input = self.input.take().unwrap();

        self.task.execute(input)
    }
} 

pub struct InputNode<In: FlowData<In>> {
    _id: In::Nodes,

    arrows_out: Vec<TaskId>,
}

impl<In: FlowData<In>> InputNode<In> {
    pub fn new(id: In::Nodes) -> Self {
        Self {
            _id: id,
            arrows_out: Vec::new(),
        }
    }
}

impl<In> FlowNode for InputNode<In>
where
    In: FlowData<In> + 'static,
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

    fn execute(&mut self, _data: &mut GraphData, _waker: &mut Waker) {
    }
}

impl<In, Out, F> Task<In, Out> for F
where
    In: 'static,
    Out: 'static,
    F: FnMut(In) -> Option<Out> + Send + 'static
{
    fn execute(&mut self, input: In) -> Option<Out> {
        self(input)    
    }
}
