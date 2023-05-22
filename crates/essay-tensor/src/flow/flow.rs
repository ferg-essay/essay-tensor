use std::{marker::PhantomData};

use super::{
    task::{Task, Tasks, TasksBuilder, NilTask, self, NodeId, TaskId}, 
    data::{FlowIn, FlowData}, dispatch::Dispatcher, pipe::{In, Out}, 
};

#[derive(Clone)]
pub struct OutputData<T: Clone + Send + 'static>(T);

pub struct Flow<I: FlowIn<I>, O: FlowIn<O>> {
    tasks: Tasks,

    input: I::Nodes,

    output_id: NodeId,
    output: In<OutputData<O>>,
}

impl<I, O> Flow<I, O>
where
    I: FlowIn<I>,
    O: FlowIn<O>
{
    pub fn builder() -> FlowBuilder<I, O> {
        FlowBuilder::new()
    }

    pub fn call(&mut self, input: I) -> Option<O> {
        let mut iter = self.iter(input);

        iter.next()
    }

    pub fn iter(&mut self, _input: I) -> FlowIter<I, O> {
        // let mut data = self.new_data();

        // In::write(&self.input, &mut data, input);

        let dispatcher = self.init(); // &mut data);

        FlowIter {
            flow: self,
            // data: data,
            dispatcher: dispatcher,
            count: 0,
        }
    }

    pub fn next(&mut self, waker: &mut Dispatcher) -> Option<O> {
        self.output.fill(waker);

        while waker.apply(&mut self.tasks) {
        }

        assert!(self.output.fill(waker));

        match self.output.next() {
            Some(OutputData(value)) => Some(value),
            None => None,
        }
    }

    fn init(&mut self) -> Dispatcher {
        let dispatcher = Dispatcher::new();

        self.output.init();
        self.tasks.init();

        dispatcher
    }
}

pub struct FlowIter<'a, I: FlowIn<I>, O: FlowIn<O>> {
    flow: &'a mut Flow<I, O>,
    dispatcher: Dispatcher,
    count: usize,
}

impl<I: FlowIn<I>, O: FlowIn<O>> Iterator for FlowIter<'_, I, O> {
    type Item = O;

    fn next(&mut self) -> Option<Self::Item> {
        self.flow.next(&mut self.dispatcher)
    }
}

impl<T: Clone + Send + 'static> OutputData<T> {
    fn unwrap(self) -> T {
        self.0
    }
}

impl<T: Clone + Send + 'static> FlowData for OutputData<T> {}

//
// Flow builder
//

pub struct FlowBuilder<I: FlowIn<I>, O: FlowIn<O>> {
    tasks: TasksBuilder,
    nil_id: TaskId<()>,
    in_nodes: I::Nodes,
    marker: PhantomData<(I, O)>,
}

impl<I: FlowIn<I>, O: FlowIn<O>> FlowBuilder<I, O> {
    fn new() -> Self {
        let mut tasks = TasksBuilder::new();

        let nil_id = tasks.push_task(Box::new(NilTask), &());

        let input_id = I::new_flow_input(&mut tasks);

        let builder = FlowBuilder {
            tasks,
            nil_id: nil_id,
            in_nodes: input_id,
            marker: Default::default(),
        };

        builder
    }

    pub fn input(&self) -> I::Nodes {
        //self.in_nodes.clone()
        todo!();
    }

    pub fn nil(&self) -> TaskId<()> {
        self.nil_id.clone()
    }

    pub fn task<I1, O1>(
        &mut self, 
        task: impl Task<I1, O1>,
        in_nodes: &I1::Nodes,
    ) -> TaskId<O1>
    where
        I1: FlowIn<I1>,
        O1: Send + Clone + 'static,
    {
        self.tasks.push_task(Box::new(task), in_nodes)
    }

    pub fn output(mut self, src_nodes: &O::Nodes) -> Flow<I, O> {
        let mut output_ids = Vec::new();

        O::node_ids(src_nodes, &mut output_ids);

        let output_task = OutputTask::<O>::new();

        let id = self.tasks.push_task(Box::new(output_task), src_nodes);
        let tail = self.tasks.push_task(Box::new(TailTask), &());

        let source = self.tasks.add_pipe(id.clone(), tail.id(), 0);

        Flow {
            tasks: self.tasks.build(),

            input: self.in_nodes,
            output: In::new(source),
            output_id: id.id(),
        }
    }
}

//
// Output task
//

struct OutputTask<O: FlowIn<O>> {
    marker: PhantomData<O>,
}

impl<O: FlowIn<O>> OutputTask<O> {
    pub(crate) fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

impl<O: FlowIn<O>> Task<O, OutputData<O>> for OutputTask<O> {
    fn next(&mut self, source: &mut O::Input) -> task::Result<Out<OutputData<O>>> {
        let value = O::next(source);

        match value {
            Out::Some(value) => {
                Ok(Out::Some(OutputData(value)))
            },
            Out::None => {
                Ok(Out::None)
            },
            Out::Pending => {
                Ok(Out::Pending)
            }
        }
    }
}

//
// tail task
//

struct TailTask;

impl Task<(), ()> for TailTask {
    fn next(&mut self, _source: &mut ()) -> task::Result<Out<()>> {
        todo!();
    }
}

#[cfg(test)]
mod test {
    use std::{sync::{Arc, Mutex}};

    use crate::flow::{
        pipe::{Out, In}
    };

    use super::{Flow};

    #[test]
    fn test_graph_nil() {
        let builder = Flow::<(), ()>::builder();
        let nil = builder.nil();
        let mut flow = builder.output(&());

        assert_eq!(flow.call(()), None);
    }

    #[test]
    fn test_graph_node() {
        let vec = Arc::new(Mutex::new(Vec::<String>::default()));
        
        let mut builder = Flow::<(), usize>::builder();
        let ptr = vec.clone();

        let node_id = builder.task::<(), usize>(move |_: &mut ()| {
            ptr.lock().unwrap().push(format!("Node[]"));
            Ok(Out::None)
        }, &());

        assert_eq!(node_id.index(), 1);

        let mut flow = builder.output(&node_id);

        assert_eq!(flow.call(()), None);
        assert_eq!(take(&vec), "Node[]");

        assert_eq!(flow.call(()), None);
        assert_eq!(take(&vec), "Node[]");
    }

    #[test]
    fn test_graph_detached_node() {
        let vec = Arc::new(Mutex::new(Vec::<String>::default()));
        
        let mut builder = Flow::<(), ()>::builder();
        let ptr = vec.clone();

        let node_id = builder.task::<(), ()>(move |_: &mut ()| {
            ptr.lock().unwrap().push(format!("Node[]"));
            Ok(Out::None)
        }, &());

        assert_eq!(node_id.index(), 1);

        // let nil = builder.nil();
        let mut flow = builder.output(&());

        assert_eq!(flow.call(()), None);
        assert_eq!(take(&vec), "");

        assert_eq!(flow.call(()), None);
        assert_eq!(take(&vec), "");
    }

    #[test]
    fn graph_sequence() {
        let vec = Arc::new(Mutex::new(Vec::<String>::default()));
        
        let mut builder = Flow::<(), bool>::builder();

        let ptr = vec.clone();
        let mut data = vec!["a".to_string(), "b".to_string()];

        let n_0 = builder.task::<(), String>(move |_: &mut ()| {
            ptr.lock().unwrap().push(format!("Node0[]"));
            match data.pop() {
                Some(v) => {
                    println!("N_0 {:?}", v);
                    Ok(Out::Some(v))
                },
                None => {
                    println!("N_0 None");
                    Ok(Out::None)
                }
            }
        }, &());

        assert_eq!(n_0.index(), 1);

        let ptr = vec.clone();
        let n_1 = builder.task::<String, bool>(move |s: &mut In<String>| {
            ptr.lock().unwrap().push(format!("Node1[{:?}]", s.next()));
            Ok(Out::None)
        }, &n_0);

        assert_eq!(n_1.index(), 2);

        let mut flow = builder.output(&n_1); // n_1);

        assert_eq!(flow.call(()), None);
        assert_eq!(take(&vec), "Node0[], Node1[b]");

        assert_eq!(flow.call(()), None);
        assert_eq!(take(&vec), "Node0[], Node1[a]");

        assert_eq!(flow.call(()), None);
        assert_eq!(take(&vec), "Node0[]");
    }

    #[test]
    fn test_graph_input() {
        let vec = Arc::new(Mutex::new(Vec::<String>::default()));
        
        let mut builder = Flow::<usize, ()>::builder();

        let ptr = vec.clone();

        let input = builder.input();
        let n_0 = builder.task::<usize, ()>(move |x: &mut In<usize>| {
            ptr.lock().unwrap().push(format!("Task[{:?}]", x.next().unwrap()));
            Ok(Out::None)
        }, &input);

        let mut flow = builder.output(&()); // n_0);

        assert_eq!(flow.call(1), Some(()));
        assert_eq!(take(&vec), "Task[1]");

        assert_eq!(flow.call(2), Some(()));
        assert_eq!(take(&vec), "Task[2]");
    }

    #[test]
    fn flow_output() {
        let vec = Arc::new(Mutex::new(Vec::<String>::default()));
        
        let mut builder = Flow::<(), usize>::builder();

        let ptr = vec.clone();

        let mut count = 2;
        let n_0 = builder.task::<(), usize>(move |_: &mut ()| {
            ptr.lock().unwrap().push(format!("Task[{}]", count));
            if count > 0 {
                count -= 1;
                Ok(Out::Some(count))
            } else {
                Ok(Out::None)
            }
        }, &());

        let mut flow = builder.output(&n_0);

        assert_eq!(flow.call(()), Some(1));
        assert_eq!(take(&vec), "Task[2]");

        assert_eq!(flow.call(()), Some(0));
        assert_eq!(take(&vec), "Task[1]");

        assert_eq!(flow.call(()), None);
        assert_eq!(take(&vec), "Task[0]");

        assert_eq!(flow.call(()), None);
        assert_eq!(take(&vec), "");
    }

    #[test]
    fn graph_input_output() {
        let vec = Arc::new(Mutex::new(Vec::<String>::default()));
        
        let mut builder = Flow::<usize, usize>::builder();

        let ptr = vec.clone();

        let input = builder.input();
        let n_0 = builder.task::<usize, usize>(move |x: &mut In<usize>| {
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
        }, &()); // builder.nil());

        let ptr = vec.clone();
        let _n_1 = builder.task::<usize, ()>(move |x: &mut In<usize>| {
            ptr.lock().unwrap().push(format!("N-1[{}]", x.next().unwrap()));
            Ok(Out::None)
        }, &n_0);

        let ptr = vec.clone();
        let n_2 = builder.task::<usize, ()>(move |x: &mut In<usize>| {
            ptr.lock().unwrap().push(format!("N-1[{}]", x.next().unwrap()));
            Ok(Out::None)
        }, &n_0);

        let mut flow = builder.output(&()); // n_2);

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
        }, &()); // builder.nil());

        let ptr = vec.clone();
        let n_2 = builder.task::<(), f32>(move |_: &mut ()| {
            ptr.lock().unwrap().push(format!("N-1[]"));
            Ok(Out::Some(10.5))
        }, &()); // builder.nil());

        let ptr = vec.clone();
        let n_2 = builder.task::<(usize, f32), ()>(move |v: &mut (In<usize>, In<f32>)| {
            ptr.lock().unwrap().push(format!("N-2[{}, {}]", v.0.next().unwrap(), v.1.next().unwrap()));
            Ok(Out::None)
        }, &(n_1, n_2));

        let mut flow = builder.output(&()); // n_2);

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
        }, &()); // builder.nil());

        let ptr = vec.clone();
        let n_2 = builder.task::<(), usize>(move |_: &mut ()| {
            ptr.lock().unwrap().push(format!("N-1[]"));
            Ok(Out::Some(10))
        }, &()); // builder.nil());

        let ptr = vec.clone();
        let n_3 = builder.task::<(), usize>(move |_: &mut ()| {
            ptr.lock().unwrap().push(format!("N-1[]"));
            Ok(Out::Some(100))
        }, &()); // builder.nil());

        let ptr = vec.clone();
        let n_4 = builder.task::<Vec<usize>, ()>(move |x: &mut Vec<In<usize>>| {
            ptr.lock().unwrap().push(format!("N-2[{:?}]", x[0].next().unwrap()));
            Ok(Out::None)
        }, &vec![n_1, n_2, n_3]);

        let mut flow = builder.output(&()); // n_4);

        assert_eq!(flow.call(()), Some(()));
        assert_eq!(take(&vec), "N-1[], N-1[], N-1[], N-2[[1, 10, 100]]");
    }

    #[test]
    fn output_with_incomplete_data() {
        let vec = Arc::new(Mutex::new(Vec::<String>::default()));
        
        let mut builder = Flow::<usize, usize>::builder();

        let ptr = vec.clone();

        let input = builder.input();
        let n_0 = builder.task::<usize, usize>(move |x: &mut In<usize>| {
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
        }, &()); // builder.nil());

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