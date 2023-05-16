use std::marker::PhantomData;

use super::{
    task::{FlowNode, Task, TaskNode, InputNode}, 
    data::{GraphData, FlowData}, 
    dispatch::{Waker, BasicDispatcher}
};


#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct TaskId(usize);

#[derive(Copy, Debug, PartialEq)]
pub struct TypedTaskId<T> {
    id: TaskId,
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

pub struct Flow<In:FlowData<In>> {
    graph: Graph,
    input: In::Nodes,
    marker: PhantomData<In>,
}

pub trait FlowNodes : Clone + 'static {
    fn add_arrows(&self, node: TaskId, graph: &mut Graph);
}

pub struct Graph {
    nodes: Vec<Box<dyn FlowNode>>,
}

pub struct FlowBuilder<In: FlowData<In>> {
    graph: Graph,
    input: In::Nodes,
    marker: PhantomData<In>,
}

impl<In: FlowData<In>> Flow<In> {
    pub fn builder() -> FlowBuilder<In> {
        FlowBuilder::new()
    }

    pub fn apply(&mut self, input: In) {
        let mut data = self.graph.new_data();

        In::write(&self.input, &mut data, input);

        self.graph.apply(&mut data)
    }
}

impl<In: FlowData<In>> FlowBuilder<In> {
    fn new() -> Self {
        let mut graph = Graph::default();

        let input_id = In::new_input(&mut graph);

        let node: InputNode<In> = InputNode::new(input_id.clone());

        graph.nodes.push(Box::new(node));

        let builder = FlowBuilder {
            graph: graph,
            input: input_id,
            marker: Default::default(),
        };

        builder
    }

    pub fn input(&self) -> In::Nodes {
        self.input.clone()
    }

    pub fn output<Out>(self, node: TypedTaskId<Out>) -> Flow<In> { // Flow<In, Out>
        todo!();
    }

    pub fn task<I, O>(
        &mut self, 
        task: impl Task<I, O>,
        input: &I::Nodes,
    ) -> TypedTaskId<O>
    where
        I: FlowData<I>,
        O: FlowData<O>,
    {
        let id = TaskId(self.graph.nodes.len());
        let typed_id = TypedTaskId::new(id);

        input.add_arrows(id, &mut self.graph);
        let node: TaskNode<I, O> = TaskNode::new(typed_id.clone(), task, input.clone());

        self.graph.nodes.push(Box::new(node));

        typed_id
    }

    pub fn build(self) -> Flow<In> {
        Flow {
            input: self.input,
            graph: self.graph,
            marker: PhantomData,
        }
    }
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn alloc_id<T: 'static>(&self) -> TypedTaskId<T> {
        let id = TaskId(self.nodes.len());

        TypedTaskId::new(id)
    }

    pub fn push_node(&mut self, node: Box<dyn FlowNode>) {
        self.nodes.push(node);
    }

    pub fn new_data(&self) -> GraphData {
        let mut data = GraphData::new();
        
        for node in &self.nodes {
            node.new_data(&mut data);
        }

        data
    }

    pub fn apply(&mut self, data: &mut GraphData) {
        let mut dispatcher = BasicDispatcher::new();
        let mut waker = Waker::new();

        for node in &mut self.nodes {
            node.init(data, &mut dispatcher, &mut waker);
        }

        while dispatcher.dispatch(self, &mut waker, data) {
            waker.wake(self, data, &mut dispatcher);
        }
    }

    pub fn node(&self, id: TaskId) -> &Box<dyn FlowNode> {
        &self.nodes[id.index()]
    }

    pub fn node_mut(&mut self, id: TaskId) -> &mut Box<dyn FlowNode> {
        &mut self.nodes[id.index()]
    }

    fn add_arrow(&mut self, src_id: TaskId, dst_id: TaskId) {
        let node = &mut self.nodes[src_id.index()];

        node.add_output_arrow(dst_id);
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self { nodes: Default::default() }
    }
}

impl TaskId {
    fn index(&self) -> usize {
        self.0
    }
}

impl<T: 'static> TypedTaskId<T> {
    fn new(id: TaskId) -> Self {
        Self {
            id,
            marker: PhantomData,
        }
    }

    #[inline]
    pub fn id(&self) -> TaskId {
        self.id
    }

    #[inline]
    pub fn index(&self) -> usize {
        self.id.index()
    }
}

impl FlowNodes for () {
    fn add_arrows(&self, _node: TaskId, _graph: &mut Graph) {
    }
}

impl<T: 'static> FlowNodes for TypedTaskId<T> {
    fn add_arrows(&self, node: TaskId, graph: &mut Graph) {
        graph.add_arrow(self.id, node);
    }
}


#[cfg(test)]
mod test {
    use std::{sync::{Arc, Mutex}};

    use super::{Flow, TaskId};

    #[test]
    fn test_graph_nil() {
        let builder = Flow::<()>::builder();
        let mut flow = builder.build();

        flow.apply(());
    }

    #[test]
    fn test_graph_node() {
        let vec = Arc::new(Mutex::new(Vec::<String>::default()));
        
        let mut builder = Flow::<()>::builder();
        let ptr = vec.clone();

        let node_id = builder.task::<(), ()>(move |_: ()| {
            ptr.lock().unwrap().push(format!("Node[]"));
            None
        }, &());

        assert_eq!(node_id.id(), TaskId(0));

        let mut flow = builder.build();

        flow.apply(());
        assert_eq!(take(&vec), "Node[]");

        flow.apply(());
        assert_eq!(take(&vec), "Node[]");
    }

    #[test]
    fn test_graph_node_pair() {
        let vec = Arc::new(Mutex::new(Vec::<String>::default()));
        
        let mut builder = Flow::<()>::builder();

        let ptr = vec.clone();
        let mut data = vec!["a".to_string(), "b".to_string()];

        let n_0 = builder.task::<(), String>(move |_| {
            ptr.lock().unwrap().push(format!("Node0[]"));
            data.pop()
        }, &());

        assert_eq!(n_0.id(), TaskId(0));

        let ptr = vec.clone();
        let n_1 = builder.task::<String, ()>(move |s| {
            println!("Execute n_1");
            ptr.lock().unwrap().push(format!("Node1[{s}]"));
            None
        }, &n_0);

        assert_eq!(n_1.id(), TaskId(1));

        let mut flow = builder.build();

        flow.apply(());
        assert_eq!(take(&vec), "Node0[], Node1[b]");

        flow.apply(());
        assert_eq!(take(&vec), "Node0[], Node1[a]");

        flow.apply(());
        assert_eq!(take(&vec), "Node0[]");
    }

    #[test]
    fn test_graph_input() {
        let vec = Arc::new(Mutex::new(Vec::<String>::default()));
        
        let mut builder = Flow::<usize>::builder();

        let ptr = vec.clone();

        let input = builder.input();
        let _n_0 = builder.task::<usize, ()>(move |x| {
            ptr.lock().unwrap().push(format!("Task[{:?}]", x));
            None
        }, &input);

        let mut flow = builder.build();

        flow.apply(1);
        assert_eq!(take(&vec), "Task[1]");

        flow.apply(2);
        assert_eq!(take(&vec), "Task[2]");
    }

    fn take(ptr: &Arc<Mutex<Vec<String>>>) -> String {
        let vec : Vec<String> = ptr.lock().unwrap().drain(..).collect();

        vec.join(", ")
    }
}