use std::marker::PhantomData;

use super::{
    task::{FlowNode, Task, TaskNode, InputTask}, 
    data::{GraphData, FlowIn}, dispatch::Dispatcher, 
};

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct TaskIdBare(usize);

#[derive(Copy, Debug, PartialEq)]
pub struct TaskId<T> {
    id: TaskIdBare,
    marker: PhantomData<T>,
}

impl<T> Clone for TaskId<T> {
    fn clone(&self) -> Self {
        Self { 
            id: self.id.clone(), 
            marker: self.marker.clone() 
        }
    }
}

pub struct Flow<In: FlowIn<In>, Out: FlowIn<Out>> {
    graph: Graph,
    input: In::Nodes,
    output: Out::Nodes,
    //marker: PhantomData<In>,
}

pub trait FlowNodes : Clone + 'static {
    fn add_arrows(&self, node: TaskIdBare, graph: &mut Graph);
}

pub struct Graph {
    nodes: Vec<Box<dyn FlowNode>>,
}

pub struct FlowBuilder<In: FlowIn<In>, Out: FlowIn<Out>> {
    graph: Graph,
    input: In::Nodes,
    marker: PhantomData<(In, Out)>,
}

impl<In, Out> Flow<In, Out>
where
    In: FlowIn<In>,
    Out: FlowIn<Out>
{
    pub fn builder() -> FlowBuilder<In, Out> {
        FlowBuilder::new()
    }

    pub fn call(&mut self, input: In) -> Option<Out> {
        let mut data = self.graph.new_data();

        In::write(&self.input, &mut data, input);

        self.graph.call(&mut data);

        if Out::is_available(&self.output, &data) {
            Some(Out::read(&self.output, &mut data))
        } else {
            None
        }
    }
}

impl<In: FlowIn<In>, Out: FlowIn<Out>> FlowBuilder<In, Out> {
    fn new() -> Self {
        let mut graph = Graph::default();

        let input_id = In::new_input(&mut graph);

        let node: InputTask<In> = InputTask::new(input_id.clone());

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

    pub fn task<I, O>(
        &mut self, 
        task: impl Task<I, O>,
        input: &I::Nodes,
    ) -> TaskId<O>
    where
        I: FlowIn<I>,
        O: Clone + 'static,
    {
        let id = TaskIdBare(self.graph.nodes.len());
        let typed_id = TaskId::new(id);

        input.add_arrows(id, &mut self.graph);
        let node: TaskNode<I, O> = TaskNode::new(typed_id.clone(), task, input.clone());

        self.graph.nodes.push(Box::new(node));

        typed_id
    }

    pub fn output(self, output: &Out::Nodes) -> Flow<In, Out> {
        Flow {
            graph: self.graph,

            input: self.input,
            output: output.clone(),
        }
    }
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn alloc_id<T: 'static>(&self) -> TaskId<T> {
        let id = TaskIdBare(self.nodes.len());

        TaskId::new(id)
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

    pub fn call(&mut self, data: &mut GraphData) {
        let mut dispatcher = Dispatcher::new();

        for node in &mut self.nodes {
            node.init(data, &mut dispatcher);
        }

        while dispatcher.dispatch(self, data) {
            dispatcher.wake(self, data);
        }
    }

    pub fn node(&self, id: TaskIdBare) -> &Box<dyn FlowNode> {
        &self.nodes[id.index()]
    }

    pub fn node_mut(&mut self, id: TaskIdBare) -> &mut Box<dyn FlowNode> {
        &mut self.nodes[id.index()]
    }

    pub fn add_arrow(&mut self, src_id: TaskIdBare, dst_id: TaskIdBare) {
        let node = &mut self.nodes[src_id.index()];

        node.add_output_arrow(dst_id);
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self { nodes: Default::default() }
    }
}

impl TaskIdBare {
    fn index(&self) -> usize {
        self.0
    }
}

impl<T: 'static> TaskId<T> {
    fn new(id: TaskIdBare) -> Self {
        Self {
            id,
            marker: PhantomData,
        }
    }

    #[inline]
    pub fn id(&self) -> TaskIdBare {
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

    use super::{Flow, TaskIdBare};

    #[test]
    fn test_graph_nil() {
        let builder = Flow::<(), ()>::builder();
        let mut flow = builder.output(&());

        flow.call(());
    }

    #[test]
    fn test_graph_node() {
        let vec = Arc::new(Mutex::new(Vec::<String>::default()));
        
        let mut builder = Flow::<(), ()>::builder();
        let ptr = vec.clone();

        let node_id = builder.task::<(), ()>(move |_: ()| {
            ptr.lock().unwrap().push(format!("Node[]"));
            None
        }, &());

        assert_eq!(node_id.id(), TaskIdBare(1));

        let mut flow = builder.output(&());

        assert_eq!(flow.call(()), Some(()));
        assert_eq!(take(&vec), "Node[]");

        assert_eq!(flow.call(()), Some(()));
        assert_eq!(take(&vec), "Node[]");
    }

    #[test]
    fn test_graph_node_pair() {
        let vec = Arc::new(Mutex::new(Vec::<String>::default()));
        
        let mut builder = Flow::<(), ()>::builder();

        let ptr = vec.clone();
        let mut data = vec!["a".to_string(), "b".to_string()];

        let n_0 = builder.task::<(), String>(move |_| {
            ptr.lock().unwrap().push(format!("Node0[]"));
            data.pop()
        }, &());

        assert_eq!(n_0.id(), TaskIdBare(1));

        let ptr = vec.clone();
        let n_1 = builder.task::<String, ()>(move |s| {
            ptr.lock().unwrap().push(format!("Node1[{s}]"));
            None
        }, &n_0);

        assert_eq!(n_1.id(), TaskIdBare(2));

        let mut flow = builder.output(&());

        assert_eq!(flow.call(()), Some(()));
        assert_eq!(take(&vec), "Node0[], Node1[b]");

        assert_eq!(flow.call(()), Some(()));
        assert_eq!(take(&vec), "Node0[], Node1[a]");

        assert_eq!(flow.call(()), Some(()));
        assert_eq!(take(&vec), "Node0[]");
    }

    #[test]
    fn test_graph_input() {
        let vec = Arc::new(Mutex::new(Vec::<String>::default()));
        
        let mut builder = Flow::<usize, ()>::builder();

        let ptr = vec.clone();

        let input = builder.input();
        let _n_0 = builder.task::<usize, ()>(move |x| {
            ptr.lock().unwrap().push(format!("Task[{:?}]", x));
            None
        }, &input);

        let mut flow = builder.output(&());

        assert_eq!(flow.call(1), Some(()));
        assert_eq!(take(&vec), "Task[1]");

        assert_eq!(flow.call(2), Some(()));
        assert_eq!(take(&vec), "Task[2]");
    }

    #[test]
    fn graph_output() {
        let vec = Arc::new(Mutex::new(Vec::<String>::default()));
        
        let mut builder = Flow::<usize, usize>::builder();

        let ptr = vec.clone();

        let input = builder.input();
        let n_0 = builder.task::<usize, usize>(move |x| {
            ptr.lock().unwrap().push(format!("Task[{:?}]", x));
            Some(x + 10)
        }, &input);

        let mut flow = builder.output(&n_0);

        assert_eq!(flow.call(1), Some(11));
        assert_eq!(take(&vec), "Task[1]");

        assert_eq!(flow.call(2), Some(12));
        assert_eq!(take(&vec), "Task[2]");
    }

    #[test]
    fn node_output_split() {
        let vec = Arc::new(Mutex::new(Vec::<String>::default()));
        
        let mut builder = Flow::<(), ()>::builder();

        let ptr = vec.clone();
        let n_0 = builder.task::<(), usize>(move |_| {
            ptr.lock().unwrap().push(format!("N-0[]"));
            Some(1)
        }, &());

        let ptr = vec.clone();
        let _n_1 = builder.task::<usize, ()>(move |x| {
            ptr.lock().unwrap().push(format!("N-1[{}]", x));
            None
        }, &n_0);

        let ptr = vec.clone();
        let _n_2 = builder.task::<usize, ()>(move |x| {
            ptr.lock().unwrap().push(format!("N-1[{}]", x));
            None
        }, &n_0);

        let mut flow = builder.output(&());

        assert_eq!(flow.call(()), Some(()));
        assert_eq!(take(&vec), "N-0[], N-1[1], N-1[1]");
    }

    #[test]
    fn node_tuple_input() {
        let vec = Arc::new(Mutex::new(Vec::<String>::default()));
        
        let mut builder = Flow::<(), ()>::builder();

        let ptr = vec.clone();
        let n_1 = builder.task::<(), usize>(move |_| {
            ptr.lock().unwrap().push(format!("N-1[]"));
            Some(1)
        }, &());

        let ptr = vec.clone();
        let n_2 = builder.task::<(), f32>(move |_| {
            ptr.lock().unwrap().push(format!("N-1[]"));
            Some(10.5)
        }, &());

        let ptr = vec.clone();
        let _n_2 = builder.task::<(usize, f32), ()>(move |(x, y)| {
            ptr.lock().unwrap().push(format!("N-2[{}, {}]", x, y));
            None
        }, &(n_1, n_2));

        let mut flow = builder.output(&());

        assert_eq!(flow.call(()), Some(()));
        assert_eq!(take(&vec), "N-1[], N-1[], N-2[1, 10.5]");
    }

    #[test]
    fn node_vec_input() {
        let vec = Arc::new(Mutex::new(Vec::<String>::default()));
        
        let mut builder = Flow::<(), ()>::builder();

        let ptr = vec.clone();
        let n_1 = builder.task::<(), usize>(move |_| {
            ptr.lock().unwrap().push(format!("N-1[]"));
            Some(1)
        }, &());

        let ptr = vec.clone();
        let n_2 = builder.task::<(), usize>(move |_| {
            ptr.lock().unwrap().push(format!("N-1[]"));
            Some(10)
        }, &());

        let ptr = vec.clone();
        let n_3 = builder.task::<(), usize>(move |_| {
            ptr.lock().unwrap().push(format!("N-1[]"));
            Some(100)
        }, &());

        let ptr = vec.clone();
        let _n_2 = builder.task::<Vec<usize>, ()>(move |x| {
            ptr.lock().unwrap().push(format!("N-2[{:?}]", &x));
            None
        }, &vec![n_1, n_2, n_3]);

        let mut flow = builder.output(&());

        assert_eq!(flow.call(()), Some(()));
        assert_eq!(take(&vec), "N-1[], N-1[], N-1[], N-2[[1, 10, 100]]");
    }

    #[test]
    fn output_with_incomplete_data() {
        let vec = Arc::new(Mutex::new(Vec::<String>::default()));
        
        let mut builder = Flow::<usize, usize>::builder();

        let ptr = vec.clone();

        let input = builder.input();
        let n_0 = builder.task::<usize, usize>(move |x| {
            ptr.lock().unwrap().push(format!("Task[{:?}]", x));
            None
        }, &input);

        let mut flow = builder.output(&n_0);

        assert_eq!(flow.call(1), None);
        assert_eq!(take(&vec), "Task[1]");

        assert_eq!(flow.call(2), None);
        assert_eq!(take(&vec), "Task[2]");
    }

    fn take(ptr: &Arc<Mutex<Vec<String>>>) -> String {
        let vec : Vec<String> = ptr.lock().unwrap().drain(..).collect();

        vec.join(", ")
    }
}