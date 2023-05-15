use std::{collections::VecDeque, marker::PhantomData, sync::Mutex, any::TypeId, ptr::NonNull, alloc::Layout, mem::{self, ManuallyDrop}, cell::UnsafeCell};
use futures::prelude::*;

use crate::Tensor;

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct NodeId(usize);

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct ArrowId(usize);

type NodeFun<In, Out> = dyn FnMut(In) -> Option<Out>;
type BoxFun<In, Out> = Box<NodeFun<In, Out>>;

#[derive(Copy, Debug, PartialEq)]
pub struct TypedNodeId<T> {
    id: NodeId,
    marker: PhantomData<T>,
}

impl<T> Clone for TypedNodeId<T> {
    fn clone(&self) -> Self {
        Self { 
            id: self.id.clone(), 
            marker: self.marker.clone() 
        }
    }
}

pub trait FlowData {
    type Item;
    type Nodes : Clone;

    fn read(nodes: &Self::Nodes, data: &mut GraphData) -> Option<Self::Item>;
    fn write(nodes: &Self::Nodes, data: &mut GraphData, value: Self::Item) -> bool;
}

pub struct FlowContext<'a> {
    vec: &'a Vec<NodeId>,
    index: usize,
}

trait Scalar {}

pub struct Nodes<T: FlowData> {
    nodes: Vec<NodeId>,
    marker: PhantomData<T>,
}

trait Node {
    fn add_output_arrow(&mut self, id: NodeId);

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

pub struct Graph {
    nodes: Vec<Box<dyn Node>>,
}

pub struct GraphData {
    nodes: Vec<RawData>,
}

trait FlowIn : Clone {}

struct TaskNode<In: FlowData<Item=In>, Out>
{
    id: TypedNodeId<Out>,

    state: NodeState,

    //arrows_in: BoxArrow<In>,
    arrows_in: In::Nodes, // BoxArrow<In>,
    arrows_out: Vec<NodeId>,

    output: VecDeque<Out>,

    inner: Mutex<TaskNodeInner<In, Out>>,
}

pub trait ArrowData : 'static {
    type Value;

    fn input(&self, graph: &mut GraphData) -> Option<Self::Value>;
}

type BoxArrow<I> = Box<dyn ArrowData<Value=I>>;

pub trait IntoArrow<K, V> {
    fn into_arrow(
        self, 
        node: NodeId, 
        graph: &mut Graph
    ) -> Box<dyn ArrowData<Value=V>>;
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

impl<In> FlowBuilder<In> {
    pub fn node2<K, I, O>(
        &mut self, 
        task: impl Task<I, O>,
        inputs: impl IntoArrow<K, I>,
    ) -> TypedNodeId<O>
    where
        K: Clone + 'static,
        I: 'static, // ArrowData<K, Value=I>,
        O: 'static,
    {
        let id = NodeId(self.graph.nodes.len());
        /*
        let input = inputs.into_arrow(id, &mut self.graph);
        let node: TaskNode<I, O> = TaskNode::new(id, task, input);

        self.graph.nodes.push(Box::new(node));
        */

        TypedNodeId::new(id)
    }

    pub fn input(&mut self) -> TypedNodeId<In> {
        todo!();
    }

    pub fn output<Out>(self, node: TypedNodeId<Out>) -> Flow<In> { // Flow<In, Out>
        todo!();
    }

    pub fn node<I: FlowData<Item=I>, O>(
        &mut self, 
        task: impl Task<I, O>,
        input: &I::Nodes,
    ) -> TypedNodeId<O>
    where
        I: 'static, // ArrowData<K, Value=I>,
        O: 'static,
    {
        let id = NodeId(self.graph.nodes.len());
        let typed_id = TypedNodeId::new(id);

        // let input = inputs.into_arrow(id, &mut self.graph);
        let node: TaskNode<I, O> = TaskNode::new(typed_id.clone(), task, input.clone());

        self.graph.nodes.push(Box::new(node));

        typed_id
    }

    /*
    pub fn t_node<I, O>(
        &mut self, 
        task: impl Task<I, O>,
        inputs: Key<I>,
    ) -> TypedNodeId<O>
    where
        I: 'static, // ArrowData<K, Value=I>,
        O: 'static,
    {
        todo!()
    }
    */

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

impl<In> Default for Flow<In> {
    fn default() -> Self {
        Self { 
            graph: Default::default(),
            marker: Default::default(),
        }
    }
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn apply(&mut self) {
        let mut dispatcher = BasicDispatcher::new();
        let mut data = GraphData::new(self.nodes.len());

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

    fn add_arrow(&mut self, src_id: NodeId, dst_id: NodeId) {
        let node = &mut self.nodes[src_id.index()];

        node.add_output_arrow(dst_id);
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
            
            node.execute(data, waker);
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
    In: FlowData<Item=In> + 'static, // ArrowData<Key, Value=In> + 'static,
    Out: 'static
{
    fn new(
        id: TypedNodeId<Out>,
        task: impl Task<In, Out>,
        input: In::Nodes, // BoxArrow<In>,
    ) -> Self {
        let inner : TaskNodeInner<In, Out> = TaskNodeInner::new(Box::new(task));

        Self {
            id: id,
            state: NodeState::WaitingIn,
            arrows_in: input,
            arrows_out: Default::default(),
            output: Default::default(),
            inner: Mutex::new(inner),
        }
    }
}

impl<In: FlowData<Item=In>, Out> Node for TaskNode<In, Out>
where
    In: 'static, // ArrowData<Value=In>,
    Out: 'static
{
    fn add_output_arrow(&mut self, id: NodeId) {
        self.arrows_out.push(id);
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

    fn complete(&mut self, dispatcher: &dyn Dispatcher) -> bool {
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

impl<T: 'static> TypedNodeId<T> {
    fn new(id: NodeId) -> Self {
        Self {
            id,
            marker: PhantomData,
        }
    }

    #[inline]
    fn id(&self) -> NodeId {
        self.id
    }

    #[inline]
    fn index(&self) -> usize {
        self.id.index()
    }
}

trait GraphDataTrait {
    fn read<T: 'static>(&mut self) -> Option<T>;
    fn write<T: 'static>(&mut self, value: T) -> bool;
}

struct SingleGraphData<T> {
    type_id: TypeId,
    value: Option<T>,
}

impl<T: 'static> SingleGraphData<T> {
    fn new() -> Self {
        Self {
            type_id: TypeId::of::<T>(),
            value: None,
        }
    }
}

impl<T> GraphDataTrait for SingleGraphData<T> {
    fn read<T1: 'static>(&mut self) -> Option<T1> {
        assert!(self.type_id == TypeId::of::<T1>());

        match self.value.take() {
            Some(value) => {
                unsafe {
                    Some(UnsafeCell::new(value).get().cast::<T1>().read())
                }
            }
            None => None,
        }
    }

    fn write<T1: 'static>(&mut self, value: T1) -> bool {
        assert!(self.type_id == TypeId::of::<T1>());
        assert!(self.value.is_none());

        let value = unsafe {
            UnsafeCell::new(value).get().cast::<T>().read()
        };

        self.value.replace(value);

        false
    }
}

impl GraphData {
    fn new(n: usize) -> Self {
        let mut vec = Vec::new();

        for _ in 0..n {
            vec.push(RawData::default());
        }

        Self {
            nodes: vec,
        }
    }

    fn read<T: 'static>(&mut self, node: &TypedNodeId<T>) -> Option<T> {
        println!("Read {:?}", node.id());
        self.nodes[node.index()].read()
    }
    
    fn write<T: 'static>(&mut self, node: &TypedNodeId<T>, data: T) -> bool {
        println!("Write {:?}", node.id());
        self.nodes[node.index()].write(data)
    }
}

pub struct RawData {
    item: Option<RawItem>,
}

impl Default for RawData {
    fn default() -> Self {
        Self {
            item: None,
        }
    }
}

struct RawItem {
    type_id: TypeId,
    data: NonNull<u8>,
}

impl RawItem {
    fn new<T: 'static>(item: T) -> RawItem {
        let layout = Layout::new::<T>();

        unsafe {
            let data = std::alloc::alloc(layout);
            let data = NonNull::new(data).unwrap();

            let mut value = ManuallyDrop::new(item);
            let source = NonNull::from(&mut *value).cast();

            std::ptr::copy_nonoverlapping::<u8>(
                source.as_ptr(), 
                data.as_ptr(),
                mem::size_of::<T>(),
            );

            Self {
                type_id: TypeId::of::<T>(),
                data: data,
            }
        }
    }

    fn read<T: 'static>(self) -> T {
        assert!(self.type_id == TypeId::of::<T>());

        unsafe { self.data.as_ptr().cast::<T>().read() }
    }
}

impl RawData {
    fn read<T: 'static>(&mut self) -> Option<T> {
        match self.item.take() {
            Some(raw) => Some(raw.read::<T>()),
            None => None,
        }
    }

    fn write<T: 'static>(&mut self, data: T) -> bool {
        assert!(self.item.is_none());
        self.item.replace(RawItem::new::<T>(data));

        false
    }
}

impl GraphData {
}

impl Waker {
    fn new() -> Self {
        Self {
            nodes: Default::default(),
        }
    }

    fn wake(
        &mut self, 
        graph: &mut Graph, 
        data: &mut GraphData, 
        dispatcher: &mut dyn Dispatcher
    ) {
        for id in self.nodes.drain(..) {
            graph.node_mut(id).update(data, dispatcher);
        }
    }

    fn complete(&mut self, node: NodeId, _data: &mut GraphData) {
        self.nodes.push(node);
        println!("Complete");
    }
}
/*
impl ArrowData<NodeId> for () {
    type Value = ();

    fn input(graph: &mut GraphData, nodes: &NodeId) -> Option<Self::Value> {
        Some(())
    }
}
*/

struct NilArrows;

impl ArrowData for NilArrows {
    type Value = ();

    fn input(&self, _graph: &mut GraphData) -> Option<Self::Value> {
        Some(())
    }
}

impl IntoArrow<(), ()> for () {
    fn into_arrow(self, _id: NodeId, _graph: &mut Graph) -> BoxArrow<()> {
        Box::new(NilArrows)
    }
}
/*
impl IntoArrow<NodeId, ()> for () {
    fn into_arrow(self) -> Box<dyn ArrowData<Value=()>> {
        todo!()
    }
}
*/
/*
impl IntoArrow<NodeId, String> for NodeId {
    fn into_arrow(self) -> Box<dyn ArrowData<Value=String>> {
        todo!()
    }
}
*/
struct SingleArrow<T> {
    node: TypedNodeId<T>,
}

impl<T> SingleArrow<T> {
    fn new(node: TypedNodeId<T>) -> Self {
        Self {
            node,
        }
    }
}

impl<T> ArrowData for SingleArrow<T> 
where
    T: 'static
{
    type Value = T;

    fn input(&self, graph: &mut GraphData) -> Option<Self::Value> {
        graph.read(&self.node)
    }
}
/*
impl<T:FlowData> IntoArrow<TypedNodeId<T>, T> for TypedNodeId<T> {
    fn into_arrow(self, id: NodeId, graph: &mut Graph) -> BoxArrow<T> {
        graph.add_arrow(self.id(), id);

        Box::new(SingleArrow::new(self))
    }
}
*/

impl FlowContext<'_> {
    fn next(&mut self) -> NodeId {
        let id = self.vec[self.index];
        self.index += 1;
        id
    }
}

impl FlowData for () {
    type Item = ();
    type Nodes = ();

    fn read(nodes: &Self::Nodes, data: &mut GraphData) -> Option<Self::Item> {
        Some(())
    }

    fn write(nodes: &Self::Nodes, data: &mut GraphData, value: Self::Item) -> bool {
        false
    }
}

impl<T:Scalar + Clone + 'static> FlowData for T {
    type Item = T;
    type Nodes = TypedNodeId<T>;

    fn read(nodes: &Self::Nodes, data: &mut GraphData) -> Option<Self::Item> {
        data.read(nodes)
    }

    fn write(nodes: &Self::Nodes, data: &mut GraphData, value: Self::Item) -> bool {
        data.write(nodes, value)
    }
}

// pub trait FlowData : Send + 'static {}

//impl Scalar for () {}
impl Scalar for String {}
impl Scalar for usize {}

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

struct TestKey<T> {
    nodes: Vec<NodeId>,
    marker: PhantomData<T>,
}

trait Param {}

impl<T> TestKey<T> {
    fn combine<Param>(keys: Param) -> TestKey<Param> {
        todo!();
    }
}

impl Param for () {
}

impl<T> Param for TypedNodeId<T> {
}

impl<S, T> Param for (TypedNodeId<T>, TypedNodeId<S>) {
}

/*
impl ArrowData<NodeId> for String {
    type Value = String;

    fn input(graph: &mut GraphData, arrows: &NodeId) -> Option<Self::Value> {
        todo!()
    }
}
*/

#[cfg(test)]
mod test {
    use std::{sync::{Arc, Mutex}};

    use crate::flow::node::{NodeId, TypedNodeId};

    use super::{Flow};

    #[test]
    fn test_graph_nil() {
        let builder = Flow::<()>::builder();
        let mut flow = builder.build();

        flow.apply(&());
    }

    #[test]
    fn test_graph_node() {
        let vec = Arc::new(Mutex::new(Vec::<String>::default()));
        
        let mut builder = Flow::<()>::builder();
        let ptr = vec.clone();

        let node_id = builder.node::<(), ()>(move |_: ()| {
            ptr.lock().unwrap().push(format!("Node[]"));
            None
        }, &());

        assert_eq!(node_id.id(), NodeId(0));

        let mut flow = builder.build();

        flow.apply(&());
        assert_eq!(take(&vec), "Node[]");

        flow.apply(&());
        assert_eq!(take(&vec), "Node[]");
    }

    #[test]
    fn test_graph_node_pair() {
        let vec = Arc::new(Mutex::new(Vec::<String>::default()));
        
        let mut builder = Flow::<()>::builder();

        let ptr = vec.clone();
        let mut data = Some("test".to_string());

        let n_0 = builder.node::<(), String>(move |_| {
            ptr.lock().unwrap().push(format!("Node0[]"));
            data.take()
        }, &());

        assert_eq!(n_0.id(), NodeId(0));

        let ptr = vec.clone();
        let n_1 = builder.node::<String, ()>(move |s| {
            println!("Execute n_1");
            ptr.lock().unwrap().push(format!("Node1[{s}]"));
            None
        }, &n_0);

        assert_eq!(n_1.id(), NodeId(1));

        let mut flow = builder.build();

        flow.apply(&());
        assert_eq!(take(&vec), "Node0[], Node1[test]");

        flow.apply(&());
        assert_eq!(take(&vec), "Node0[], Node1[test]");
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