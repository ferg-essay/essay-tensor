use std::marker::PhantomData;

use super::{
    task::{TaskOuter, Task, TaskNode, InputTask, NilTask}, 
    data::{GraphData, FlowIn}, dispatch::Dispatcher, graph::{TaskId, TaskIdBare, Graph}, 
};

pub struct Flow<In: FlowIn<In>, Out: FlowIn<Out>> {
    graph: Graph,
    tasks: TaskGraph,

    input: In::Nodes,
    output: Out::Nodes,
    //marker: PhantomData<In>,
}

pub struct TaskGraph {
    task_outer: Vec<Box<dyn TaskOuter>>,
}

pub struct FlowBuilder<In: FlowIn<In>, Out: FlowIn<Out>> {
    graph: Graph,
    tasks: TaskGraph,
    nil_id: TaskId<()>,
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
        let mut iter = self.iter(input);

        iter.next()
        /*
        let mut data = self.new_data();

        In::write(&self.input, &mut data, input);

        let mut dispatcher = self.init(&mut data);

        self.call_inner(&mut dispatcher, &mut data);

        if Out::is_available(&self.output, &data) {
            Some(Out::read(&self.output, &mut data))
        } else {
            None
        }
        */
    }

    pub fn iter(&mut self, input: In) -> FlowIter<In, Out> {
        let mut data = self.new_data();

        In::write(&self.input, &mut data, input);

        let dispatcher = self.init(&mut data);

        FlowIter {
            flow: self,
            data: data,
            dispatcher: dispatcher,
        }
    }

    pub fn next(&mut self, dispatcher: &mut Dispatcher, data: &mut GraphData) -> Option<Out> {
        self.graph.wake::<Out>(self.output.clone(), &mut self.tasks, dispatcher, data);
        // self.graph.node_mut(self.output.id()).require_output();

        while dispatcher.dispatch(&mut self.tasks, data) {
            dispatcher.wake(&mut self.tasks, data);
        }

        if Out::is_available(&self.output, data) {
            Some(Out::read(&self.output, data))
        } else {
            None
        }
    }

    fn new_data(&self) -> GraphData {
        let mut data = GraphData::new();
        
        for node in self.tasks.tasks() {
            node.new_data(&mut data);
        }

        data
    }

    fn init(&mut self, data: &mut GraphData) -> Dispatcher {
        let mut dispatcher = Dispatcher::new();

        for node in self.tasks.tasks_mut() {
            node.init(data, &mut dispatcher);
        }

        dispatcher
    }

    fn call_inner(&mut self, dispatcher: &mut Dispatcher, data: &mut GraphData) {
        while dispatcher.dispatch(&mut self.tasks, data) {
            dispatcher.wake(&mut self.tasks, data);
        }
    }
    /*
    pub fn node(&self, id: TaskIdBare) -> &Box<dyn TaskOuter> {
        &self.task_outer[id.index()]
    }

    pub fn node_mut(&mut self, id: TaskIdBare) -> &mut Box<dyn TaskOuter> {
        &mut self.task_outer[id.index()]
    }
    */
}

pub struct FlowIter<'a, In: FlowIn<In>, Out: FlowIn<Out>> {
    flow: &'a mut Flow<In, Out>,
    data: GraphData,
    dispatcher: Dispatcher,
}

impl<In: FlowIn<In>, Out: FlowIn<Out>> Iterator for FlowIter<'_, In, Out> {
    type Item = Out;

    fn next(&mut self) -> Option<Self::Item> {
        self.flow.next(&mut self.dispatcher, &mut self.data)
    }
}

impl<In: FlowIn<In>, Out: FlowIn<Out>> FlowBuilder<In, Out> {
    fn new() -> Self {
        let mut graph = Graph::new();
        let mut tasks = TaskGraph::default();

        let nil_id = graph.push_input::<()>();
        let node = NilTask::new(); // nil_id.clone());
        tasks.task_outer.push(Box::new(node));

        let input_id = In::new_input(&mut graph, &mut tasks);

        let node: InputTask<In> = InputTask::new(input_id.clone());
        tasks.task_outer.push(Box::new(node));

        let builder = FlowBuilder {
            graph,
            tasks,
            nil_id: nil_id,
            input: input_id,
            marker: Default::default(),
        };

        builder
    }

    pub fn input(&self) -> In::Nodes {
        self.input.clone()
    }

    pub fn nil(&self) -> TaskId<()> {
        self.nil_id.clone()
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
        let id = self.graph.push::<I, O>(input.clone());

        let task: TaskNode<I, O> = TaskNode::new(id.clone(), task, input.clone());

        self.tasks.task_outer.push(Box::new(task));

        id
    }

    pub fn output(self, output: &Out::Nodes) -> Flow<In, Out> {
        Flow {
            graph: self.graph,
            tasks: self.tasks,

            input: self.input,
            output: output.clone(),
        }
    }
}

impl TaskGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_node(&mut self, node: Box<dyn TaskOuter>) {
        self.task_outer.push(node);
    }

    fn wake(
        &mut self,
        id: TaskIdBare,
        graph: &mut TaskGraph,
        dispatcher: &mut Dispatcher,
        data: &mut GraphData,
    ) -> bool {
        let task = &mut self.task_outer[id.index()];

        todo!()
    }

    /*
    pub fn new_data(&self) -> GraphData {
        let mut data = GraphData::new();
        
        for node in &self.task_outer {
            node.new_data(&mut data);
        }

        data
    }

    pub fn init(&mut self, data: &mut GraphData) -> Dispatcher {
        let mut dispatcher = Dispatcher::new();

        for node in &mut self.task_outer {
            node.init(data, &mut dispatcher);
        }

        dispatcher
    }

    pub fn call(&mut self, dispatcher: &mut Dispatcher, data: &mut GraphData) {
        while dispatcher.dispatch(self, data) {
            dispatcher.wake(self, data);
        }
    }
    */

    pub fn node(&self, id: TaskIdBare) -> &Box<dyn TaskOuter> {
        &self.task_outer[id.index()]
    }

    pub fn tasks(&self) -> &Vec<Box<dyn TaskOuter>> {
        &self.task_outer
    }

    pub fn tasks_mut(&mut self) -> &mut Vec<Box<dyn TaskOuter>> {
        &mut self.task_outer
    }

    pub fn node_mut(&mut self, id: TaskIdBare) -> &mut Box<dyn TaskOuter> {
        &mut self.task_outer[id.index()]
    }

    pub fn add_arrow(&mut self, src_id: TaskIdBare, dst_id: TaskIdBare) {
        let node = &mut self.task_outer[src_id.index()];

        node.add_output_arrow(dst_id);
    }
}

impl Default for TaskGraph {
    fn default() -> Self {
        Self { task_outer: Default::default() }
    }
}

#[cfg(test)]
mod test {
    use std::{sync::{Arc, Mutex}};

    use crate::flow::{data::Out, task::Source};

    use super::{Flow};

    #[test]
    fn test_graph_nil() {
        let builder = Flow::<(), ()>::builder();
        let nil = builder.nil();
        let mut flow = builder.output(&nil);

        assert_eq!(flow.call(()), None);
    }

    #[test]
    fn test_graph_node() {
        let vec = Arc::new(Mutex::new(Vec::<String>::default()));
        
        let mut builder = Flow::<(), ()>::builder();
        let ptr = vec.clone();

        let node_id = builder.task::<(), ()>(move |_: &mut ()| {
            ptr.lock().unwrap().push(format!("Node[]"));
            Ok(Out::None)
        }, &builder.nil());

        assert_eq!(node_id.index(), 3);

        let mut flow = builder.output(&node_id);

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

        let n_0 = builder.task::<(), String>(move |_: &mut ()| {
            ptr.lock().unwrap().push(format!("Node0[]"));
            match data.pop() {
                Some(v) => Ok(Out::Some(v)),
                None => Ok(Out::None)
            }
        }, &builder.nil());

        assert_eq!(n_0.index(), 1);

        let ptr = vec.clone();
        let n_1 = builder.task::<String, ()>(move |s: &mut Source<String>| {
            ptr.lock().unwrap().push(format!("Node1[{}]", s.next().unwrap()));
            Ok(Out::None)
        }, &n_0);

        assert_eq!(n_1.index(), 2);

        let mut flow = builder.output(&n_1);

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
        let n_0 = builder.task::<usize, ()>(move |x: &mut Source<usize>| {
            ptr.lock().unwrap().push(format!("Task[{:?}]", x.next().unwrap()));
            Ok(Out::None)
        }, &input);

        let mut flow = builder.output(&n_0);

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
        let n_0 = builder.task::<usize, usize>(move |x: &mut Source<usize>| {
            let x_v = x.next().unwrap();
            ptr.lock().unwrap().push(format!("Task[{:?}]", x_v));
            Ok(Out::Some(x_v + 10))
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
        let n_0 = builder.task::<(), usize>(move |_: &mut ()| {
            ptr.lock().unwrap().push(format!("N-0[]"));
            Ok(Out::Some(1))
        }, &builder.nil());

        let ptr = vec.clone();
        let _n_1 = builder.task::<usize, ()>(move |x: &mut Source<usize>| {
            ptr.lock().unwrap().push(format!("N-1[{}]", x.next().unwrap()));
            Ok(Out::None)
        }, &n_0);

        let ptr = vec.clone();
        let n_2 = builder.task::<usize, ()>(move |x: &mut Source<usize>| {
            ptr.lock().unwrap().push(format!("N-1[{}]", x.next().unwrap()));
            Ok(Out::None)
        }, &n_0);

        let mut flow = builder.output(&n_2);

        assert_eq!(flow.call(()), Some(()));
        assert_eq!(take(&vec), "N-0[], N-1[1], N-1[1]");
    }

    #[test]
    fn node_tuple_input() {
        let vec = Arc::new(Mutex::new(Vec::<String>::default()));
        
        let mut builder = Flow::<(), ()>::builder();

        let ptr = vec.clone();
        let n_1 = builder.task::<(), usize>(move |_: &mut ()| {
            ptr.lock().unwrap().push(format!("N-1[]"));
            Ok(Out::Some(1))
        }, &builder.nil());

        let ptr = vec.clone();
        let n_2 = builder.task::<(), f32>(move |_: &mut ()| {
            ptr.lock().unwrap().push(format!("N-1[]"));
            Ok(Out::Some(10.5))
        }, &builder.nil());

        let ptr = vec.clone();
        let n_2 = builder.task::<(usize, f32), ()>(move |v: &mut (Source<usize>, Source<f32>)| {
            ptr.lock().unwrap().push(format!("N-2[{}, {}]", v.0.next().unwrap(), v.1.next().unwrap()));
            Ok(Out::None)
        }, &(n_1, n_2));

        let mut flow = builder.output(&n_2);

        assert_eq!(flow.call(()), Some(()));
        assert_eq!(take(&vec), "N-1[], N-1[], N-2[1, 10.5]");
    }

    #[test]
    fn node_vec_input() {
        let vec = Arc::new(Mutex::new(Vec::<String>::default()));
        
        let mut builder = Flow::<(), ()>::builder();

        let ptr = vec.clone();
        let n_1 = builder.task::<(), usize>(move |_: &mut ()| {
            ptr.lock().unwrap().push(format!("N-1[]"));
            Ok(Out::Some(1))
        }, &builder.nil());

        let ptr = vec.clone();
        let n_2 = builder.task::<(), usize>(move |_: &mut ()| {
            ptr.lock().unwrap().push(format!("N-1[]"));
            Ok(Out::Some(10))
        }, &builder.nil());

        let ptr = vec.clone();
        let n_3 = builder.task::<(), usize>(move |_: &mut ()| {
            ptr.lock().unwrap().push(format!("N-1[]"));
            Ok(Out::Some(100))
        }, &builder.nil());

        let ptr = vec.clone();
        let n_4 = builder.task::<Vec<usize>, ()>(move |x: &mut Vec<Source<usize>>| {
            ptr.lock().unwrap().push(format!("N-2[{:?}]", x[0].next().unwrap()));
            Ok(Out::None)
        }, &vec![n_1, n_2, n_3]);

        let mut flow = builder.output(&n_4);

        assert_eq!(flow.call(()), Some(()));
        assert_eq!(take(&vec), "N-1[], N-1[], N-1[], N-2[[1, 10, 100]]");
    }

    #[test]
    fn output_with_incomplete_data() {
        let vec = Arc::new(Mutex::new(Vec::<String>::default()));
        
        let mut builder = Flow::<usize, usize>::builder();

        let ptr = vec.clone();

        let input = builder.input();
        let n_0 = builder.task::<usize, usize>(move |x: &mut Source<usize>| {
            ptr.lock().unwrap().push(format!("Task[{:?}]", x.next().unwrap()));
            Ok(Out::None)
        }, &input);

        let mut flow = builder.output(&n_0);

        assert_eq!(flow.call(1), None);
        assert_eq!(take(&vec), "Task[1]");

        assert_eq!(flow.call(2), None);
        assert_eq!(take(&vec), "Task[2]");
    }

    #[test]
    fn graph_iter() {
        let vec = Arc::new(Mutex::new(Vec::<String>::default()));
        
        let mut builder = Flow::<(), usize>::builder();
        let ptr = vec.clone();

        let n_0 = builder.task::<(), usize>(move |_: &mut ()| {
            ptr.lock().unwrap().push(format!("Node[]"));
            Ok(Out::Some(1))
        }, &builder.nil());

        assert_eq!(n_0.index(), 1);

        let mut flow = builder.output(&n_0);

        let mut iter = flow.iter(());

        assert_eq!(iter.next(), Some(1));
        assert_eq!(take(&vec), "Node[]");
    }

    fn take(ptr: &Arc<Mutex<Vec<String>>>) -> String {
        let vec : Vec<String> = ptr.lock().unwrap().drain(..).collect();

        vec.join(", ")
    }
}