use std::{collections::VecDeque, marker::PhantomData, sync::Mutex};
use futures::prelude::*;

use crate::Tensor;

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct NodeId(usize);

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct ArrowId(usize);

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

trait Dispatcher {
    fn spawn(&self, node: NodeId);
}

trait Waker {
    fn complete(&self, node: NodeId);
}

trait Node {
    fn init(&mut self, dispatcher: &dyn Dispatcher);
    fn update(&mut self, dispatcher: &dyn Dispatcher);
    fn complete(&mut self, dispatcher: &dyn Dispatcher) -> bool;

    fn execute(&mut self, waker: &dyn Waker);
}

trait NodeInner {
    fn execute(&mut self, waker: &dyn Waker);
}

pub struct FunNode<In, Out, F>
where
    F: Fn(&[In]) -> Out + Sized,
{
    id: NodeId,

    state: NodeState,

    arrows_in: ArrowsIn<In>,
    arrows_out: ArrowsOut<Out>,

    output: VecDeque<Out>,

    inner: Mutex<FunNodeInner<In, Out, F>>,
}

impl<In, Out, F> Node for FunNode<In, Out, F>
where
    F: Fn(&[In]) -> Out + Sized,
{
    fn init(&mut self, dispatcher: &dyn Dispatcher) {
        todo!()
    }

    fn update(&mut self, dispatcher: &dyn Dispatcher) {
        todo!()
    }

    fn complete(&mut self, dispatcher: &dyn Dispatcher) -> bool {
        todo!()
    }

    fn execute(&mut self, waker: &dyn Waker) {
        todo!()
    }
}

struct FunNodeInner<In, Out, F> {
    fun: F,
    input: Option<In>,
    output: Option<Out>,
}

struct ArrowsIn<In> {
    nodes: Vec<NodeId>,

    marker: PhantomData<In>,
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

pub struct Graph {
    nodes: Vec<Box<dyn Node>>,
}

trait FlowIn : Clone {}

struct DispatcherImpl<'a> {
    graph: &'a mut Graph,
}

impl<'a> DispatcherImpl<'a> {
    fn new(graph: &'a mut Graph) -> Self {
        Self {
            graph,
        }
    }
}

pub struct Flow<In> {
    graph: Graph,
    marker: PhantomData<In>,
}

impl<In> Default for Flow<In> {
    fn default() -> Self {
        Self { 
            graph: Default::default(),
            marker: Default::default(),
        }
    }
}

impl<In> Flow<In> {
    pub fn apply(&mut self, input: &In) {
        self.graph.apply()
    }
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn apply(&mut self) {
        println!("Graph");
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self { nodes: Default::default() }
    }
}

#[cfg(test)]
mod test {
    use super::{Flow};

    #[test]
    fn test_graph_nil() {
        let mut flow = Flow::<i32>::default();
        flow.apply(&1);
    }

    #[test]
    fn test_graph_input() {
        let mut flow = Flow::<i32>::default();
        flow.apply(&1);
    }
}