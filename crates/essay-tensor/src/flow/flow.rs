use std::marker::PhantomData;

use super::{task::{Node, Task, TaskNode}, data::{GraphData, FlowData}};


#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct NodeId(usize);

#[derive(Copy, Debug, PartialEq)]
pub struct TypedTaskId<T> {
    id: NodeId,
    marker: PhantomData<T>,
}

impl<T> Clone for TypedTaskId<T> {
    fn clone(&self) -> Self {
        Self { 
            id: self.id.clone(), 
            marker: self.marker.clone() 
        }
    }
}

pub struct Flow<In> {
    graph: Graph,
    marker: PhantomData<In>,
}

pub struct Graph {
    nodes: Vec<Box<dyn Node>>,
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

pub trait Dispatcher {
    fn spawn(&mut self, node: NodeId);
}

pub struct Waker {
    nodes: Vec<NodeId>,
}

impl<In: FlowData<In>> FlowBuilder<In> {
    pub fn input(&mut self) -> In::Nodes {
        todo!();
    }

    pub fn output<Out>(self, node: TypedTaskId<Out>) -> Flow<In> { // Flow<In, Out>
        todo!();
    }

    pub fn node<I: FlowData<I>, O: FlowData<O>>(
        &mut self, 
        task: impl Task<I, O>,
        input: &I::Nodes,
    ) -> TypedTaskId<O>
    where
        I: 'static, // ArrowData<K, Value=I>,
        O: 'static,
    {
        let id = NodeId(self.graph.nodes.len());
        let typed_id = TypedTaskId::new(id);

        // let input = inputs.into_arrow(id, &mut self.graph);
        let node: TaskNode<I, O> = TaskNode::new(typed_id.clone(), task, input.clone());

        self.graph.nodes.push(Box::new(node));

        typed_id
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
        let mut data = GraphData::new(&self.nodes);

        for node in &mut self.nodes {
            node.init(&mut data, &mut dispatcher);
        }

        let mut waker = Waker::new();
        while dispatcher.dispatch(self, &mut waker, &mut data) {
            waker.wake(self, &mut data, &mut dispatcher);
        }
    }

    pub fn node(&self, id: NodeId) -> &Box<dyn Node> {
        &self.nodes[id.index()]
    }

    pub fn node_mut(&mut self, id: NodeId) -> &mut Box<dyn Node> {
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

    pub fn complete(&mut self, node: NodeId, _data: &mut GraphData) {
        self.nodes.push(node);
        println!("Complete");
    }
}

impl NodeId {
    fn index(&self) -> usize {
        self.0
    }
}

impl<T: 'static> TypedTaskId<T> {
    fn new(id: NodeId) -> Self {
        Self {
            id,
            marker: PhantomData,
        }
    }

    #[inline]
    pub fn id(&self) -> NodeId {
        self.id
    }

    #[inline]
    pub fn index(&self) -> usize {
        self.id.index()
    }
}


#[cfg(test)]
mod test {
    use std::{sync::{Arc, Mutex}};

    use super::{Flow, NodeId};

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