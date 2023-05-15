use std::{collections::VecDeque, marker::PhantomData, sync::Mutex};
use futures::prelude::*;

use crate::Tensor;

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct NodeId(usize);

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct ArrowId(usize);

type NodeFun<In, Out> = dyn FnMut(In) -> Option<Out>;
type BoxFun<In, Out> = Box<NodeFun<In, Out>>;

pub trait Task<In, Out> : Send + 'static
where
    In: 'static,
    Out: 'static,
{
    fn execute(&mut self, input: In) -> Option<Out>;
}

type BoxTask<In, Out> = Box<dyn Task<In, Out>>;

pub struct Graph {
    nodes: Vec<Box<dyn Node>>,
}

trait FlowIn : Clone {}

trait NodeInner {
    fn execute(&mut self, waker: &Waker);
}

struct TaskNode<In, Out>
{
    id: NodeId,

    state: NodeState,

    arrows_in: Vec<NodeId>,
    arrows_out: Vec<NodeId>,

    output: VecDeque<Out>,

    inner: Mutex<TaskNodeInner<In, Out>>,
}

pub trait ArrowData : 'static {
    type Item;

    fn input(graph: &mut GraphData, arrows: &Vec<NodeId>) -> Option<Self::Item>;
}

struct TaskNodeInner<In, Out>
{
    task: BoxTask<In, Out>,
    input: Option<In>,
    output: Option<Out>,
}

struct ArrowsIn<In> {
    nodes: Vec<NodeId>,

    marker: PhantomData<In>,
}

pub struct Flow<In> {
    graph: Graph,
    marker: PhantomData<In>,
}

pub struct FlowBuilder<In> {
    graph: Graph,
    marker: PhantomData<In>,
}

struct BasicDispatcher {
    ready: Vec<NodeId>,
}

pub struct DispatcherImpl<'a> {
    graph: &'a mut Graph,
}

trait Node {
    fn init(&mut self, data: &mut GraphData, dispatcher: &mut dyn Dispatcher);
    fn update(
        &self, 
        graph: &Graph, 
        data: &mut GraphData, 
        dispatcher: &mut dyn Dispatcher
    );
    fn complete(&mut self, dispatcher: &dyn Dispatcher) -> bool;

    fn execute(&mut self, waker: &mut Waker);
}

enum NodeState {
    Active, // currently dispatched

    WaitingIn, // waiting for input
    WaitingOut, // waiting for output to clear (backpressure)
    WaitingInOut, // waiting for both input and output

    Complete,
}

enum NodeAction {
    None,
    Start,
}

struct ArrowsOut<Out> {
    nodes: Vec<NodeId>,

    marker: PhantomData<Out>,
}


struct Arrows {
    // fn pop(&self, graph: &mut Graph) -> T;
}

pub struct Arrow<T> {
    id: ArrowId,

    src: NodeId,
    dst: NodeId,

    queue: VecDeque<T>,
}

trait Dispatcher {
    fn spawn(&mut self, node: NodeId);
}

pub struct Waker {
    nodes: Vec<NodeId>,
}

impl<In> Default for Flow<In> {
    fn default() -> Self {
        Self { 
            graph: Default::default(),
            marker: Default::default(),
        }
    }
}

impl<In> FlowBuilder<In> {
    pub fn node<I, O>(
        &mut self, 
        task: impl Task<I, O>,
        inputs: &[NodeId]
    ) -> NodeId
    where
        I: ArrowData<Item=I>,
        O: 'static,
    {
        let node: TaskNode<I, O> = TaskNode::new(task);
        let id = node.id;

        self.graph.nodes.push(Box::new(node));

        id
    }

    pub fn build(self) -> Flow<In> {
        Flow {
            graph: self.graph,
            marker: PhantomData,
        }
    }
}

impl<In> Flow<In> {
    pub fn builder() -> FlowBuilder<In> {
        let builder = FlowBuilder::<In> {
            graph: Default::default(),
            marker: Default::default(),
        };

        builder
    }

    pub fn apply(&mut self, input: &In) {
        self.graph.apply()
    }
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn apply(&mut self) {
        let mut dispatcher = BasicDispatcher::new();
        let mut data = GraphData {};

        for node in &mut self.nodes {
            node.init(&mut data, &mut dispatcher);
        }

        let mut waker = Waker::new();
        while dispatcher.dispatch(self, &mut waker, &mut data) {
            waker.wake(self, &mut data, &mut dispatcher);
        }
    }

    fn node(&self, id: NodeId) -> &Box<dyn Node> {
        &self.nodes[id.index()]
    }

    fn node_mut(&mut self, id: NodeId) -> &mut Box<dyn Node> {
        &mut self.nodes[id.index()]
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self { nodes: Default::default() }
    }
}

impl<'a> DispatcherImpl<'a> {
    fn new(graph: &'a mut Graph) -> Self {
        Self {
            graph,
        }
    }
}

impl BasicDispatcher {
    fn new() -> Self {
        Self {
            ready: Default::default(),
        }
    }

    fn dispatch(
        &mut self, 
        graph: &mut Graph, 
        waker: &mut Waker,
        data: &mut GraphData
    ) -> bool {
        let mut is_active = false;

        while let Some(id) = self.ready.pop() {
            is_active = true;

            let node = graph.node_mut(id);
            
            node.execute(waker);
        }

        is_active
    }
}

impl Dispatcher for BasicDispatcher {
    fn spawn(&mut self, node: NodeId) {
        self.ready.push(node);
        println!("Spawn: {:?}", node);
    }
}

impl<In, Out> TaskNode<In, Out>
where
    In: ArrowData<Item=In> + 'static,
    Out: 'static
{
    fn new(
        task: impl Task<In, Out>
    ) -> Self {
        let inner : TaskNodeInner<In, Out> = TaskNodeInner::new(Box::new(task));

        Self {
            id: NodeId(0),
            state: NodeState::WaitingIn,
            arrows_in: Default::default(),
            arrows_out: Default::default(),
            output: Default::default(),
            inner: Mutex::new(inner),
        }
    }
}

impl<In, Out> Node for TaskNode<In, Out>
where
    In: ArrowData<Item=In>,
    Out: 'static
{
    fn init(&mut self, data: &mut GraphData, dispatcher: &mut dyn Dispatcher) {
        self.state = NodeState::WaitingIn;

        if let Some(input) = In::input(data, &self.arrows_in)  {
            self.state = NodeState::Active;
            self.inner.lock().unwrap().input.replace(input);

            dispatcher.spawn(self.id);
        }
    }

    fn update(
        &self, 
        graph: &Graph, 
        data: &mut GraphData, 
        dispatcher: &mut dyn Dispatcher
    ) {
        println!("Node update");
        todo!()
    }

    fn complete(&mut self, dispatcher: &dyn Dispatcher) -> bool {
        todo!()
    }

    fn execute(&mut self, waker: &mut Waker) {
        match self.inner.lock().unwrap().execute() {
            Some(out) => {
                self.output.push_back(out);

                // TODO: allow multi-buffer
                self.state = NodeState::WaitingOut;

                for node in &self.arrows_out {
                    waker.complete(*node);
                }
            }
            None => {
                self.state = NodeState::Complete;
            }
        }
    }
}

impl<In, Out> TaskNodeInner<In, Out>
where
    In: 'static,
    Out: 'static
{
    fn new(task: BoxTask<In, Out>) -> Self {
        Self {
            task: task,
            input: None,
            output: None,
        }
    }

    fn execute(&mut self) -> Option<Out> {
        let input = self.input.take().unwrap();

        self.task.execute(input)
    }
} 

impl ArrowsIn<()> {
    fn empty() -> ArrowsIn<()> {
        Self {
            nodes: Default::default(),
            marker: PhantomData,
        }
    }

    fn input(&self, graph: &mut Graph) -> Option<()> {
        if self.nodes.len() == 0 {
            Some(())
        } else {
            None
        }
    }
}

impl NodeId {
    fn index(&self) -> usize {
        self.0
    }
}

pub struct GraphData {

}

impl Waker {
    fn new() -> Self {
        Self {
            nodes: Default::default(),
        }
    }

    fn wake(
        &mut self, 
        graph: &Graph, 
        data: &mut GraphData, 
        dispatcher: &mut dyn Dispatcher
    ) {
        for id in self.nodes.drain(..) {
            graph.node(id).update(graph, data, dispatcher);
        }
    }

    fn complete(&self, node: NodeId) {
        println!("Complete");
    }
}

impl ArrowData for () {
    type Item = ();

    fn input(graph: &mut GraphData, nodes: &Vec<NodeId>) -> Option<Self::Item> {
        Some(())
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

#[cfg(test)]
mod test {
    use std::{rc::Rc, cell::RefCell, sync::{Arc, Mutex}};

    use crate::flow::node::NodeId;

    use super::{Flow};

    #[test]
    fn test_graph_nil() {
        let mut builder = Flow::<()>::builder();
        let mut flow = builder.build();

        flow.apply(&());
    }

    #[test]
    fn test_graph_node() {
        let vec = Arc::new(Mutex::new(Vec::<String>::default()));
        
        let mut builder = Flow::<()>::builder();
        let ptr = vec.clone();
        let node_id = builder.node::<(), ()>(move |_| {
            ptr.lock().unwrap().push(format!("Node[]"));
            None
        }, &[]);

        assert_eq!(node_id, NodeId(0));

        let mut flow = builder.build();

        flow.apply(&());
        assert_eq!(take(&vec), "Node[]");

        flow.apply(&());
        assert_eq!(take(&vec), "Node[]");
    }

    #[test]
    fn test_graph_input() {
        let mut flow = Flow::<i32>::default();
        flow.apply(&1);
    }

    fn take(ptr: &Arc<Mutex<Vec<String>>>) -> String {
        let vec : Vec<String> = ptr.lock().unwrap().drain(..).collect();

        vec.join(", ")
    }
}