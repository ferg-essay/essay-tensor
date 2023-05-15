use std::{sync::Mutex};

use super::{data::{FlowData, GraphData, RawData}, flow::{Waker, Dispatcher, NodeId, TypedTaskId}};

pub trait Node {
    fn add_output_arrow(&mut self, id: NodeId);

    fn new_data(&self) -> RawData;

    fn init(&mut self, data: &mut GraphData, dispatcher: &mut dyn Dispatcher);

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

pub trait Task<In, Out> : Send + 'static
where
    In: 'static,
    Out: 'static,
{
    fn execute(&mut self, input: In) -> Option<Out>;
}

type BoxTask<In, Out> = Box<dyn Task<In, Out>>;

pub struct TaskNode<In: FlowData<In>, Out>
{
    id: TypedTaskId<Out>,

    state: NodeState,

    arrows_in: In::Nodes,
    arrows_out: Vec<NodeId>,

    inner: Mutex<TaskInner<In, Out>>,
}

struct TaskInner<In, Out>
{
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

impl<In, Out> Node for TaskNode<In, Out>
where
    In: FlowData<In> + 'static, // ArrowData<Value=In>,
    Out: FlowData<Out> + 'static
{
    fn add_output_arrow(&mut self, id: NodeId) {
        self.arrows_out.push(id);
    }

    fn new_data(&self) -> RawData {
        RawData::new::<Out>()
    }

    fn init(&mut self, data: &mut GraphData, dispatcher: &mut dyn Dispatcher) {
        self.state = NodeState::WaitingIn;

        if let Some(input) = In::read(&self.arrows_in, data) {
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
                println!("Update-WaitingIn {:?}", self.id.id());
                if let Some(input) = In::read(&self.arrows_in, data) {
                    println!("  Some {:?}", self.id.id());
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
                println!("Task -> Some");
                // self.output.push_back(out);
                data.write(&self.id, out);

                // TODO: allow multi-buffer
                self.state = NodeState::WaitingOut;

                for node in &self.arrows_out {
                    waker.complete(*node, data);
                }
            }
            None => {
                println!("Task -> None");
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
