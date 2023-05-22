use std::{marker::PhantomData, sync::{Arc, Mutex}};

use crate::flow::dispatch::{DispatcherInner, FlowThreads};

use super::{
    source::{Source, Sources, SourcesBuilder, NilTask, self, NodeId, SourceId, SourcesInner, SourcesOuter}, 
    data::{FlowIn, FlowData}, dispatch::{DispatcherOuter, Dispatcher}, pipe::{In, Out}, thread_pool::ThreadPool, 
};

pub struct Flow<I: FlowIn<I>, O: FlowIn<O>> {
    flow: Box<dyn FlowTrait<I, O>>,
}

impl<I, O> Flow<I, O>
where
    I: FlowIn<I>,
    O: FlowIn<O>
{
    pub fn builder() -> FlowBuilder<I, O> {
        FlowBuilder::new()
    }

    fn new(flow: impl FlowTrait<I, O>) -> Self {
        Self {
            flow: Box::new(flow)
        }
    }

    pub fn call(&mut self, input: I) -> Option<O> {
        self.flow.call(input)
    }

    pub fn iter(&mut self, input: I) -> FlowIter<I, O> {
        self.flow.iter(input)
    }
}

pub trait FlowTrait<I: FlowIn<I>, O: FlowIn<O>> : 'static {
    fn call(&mut self, input: I) -> Option<O>;

    fn iter(&mut self, _input: I) -> FlowIter<I, O>;
}

//
// FlowPool
//

pub struct FlowPool<I: FlowIn<I>, O: FlowIn<O>> {
    pool: FlowThreads,
    sources_inner: Arc<SourcesInner>,

    input_ids: I::Nodes,

    output: SharedOutput<O>,
}

impl<I, O> FlowPool<I, O>
where
    I: FlowIn<I>,
    O: FlowIn<O>
{
    fn new(
        sources_outer: SourcesOuter,
        sources_inner: SourcesInner,
        input_ids: I::Nodes,
        output: SharedOutput<O>,
        tail_id: NodeId,
    ) -> Flow<I, O> {
        let sources_inner = Arc::new(sources_inner);

        let pool = FlowThreads::new(tail_id, sources_outer, sources_inner.clone());
        
        Flow::new(Self {
            pool,
            sources_inner,
            input_ids,
            output,
        })
    }
}

impl<I, O> FlowTrait<I, O> for FlowPool<I, O>
where
    I: FlowIn<I>,
    O: FlowIn<O>
{
    fn call(&mut self, input: I) -> Option<O> {
        self.output.take();

        self.sources_inner.init();

        self.pool.init().unwrap();

        match self.output.take() {
            Some(data) => Some(data),
            None => None,
        }
    }

    fn iter(&mut self, _input: I) -> FlowIter<I, O> {
        todo!();
    }
}

// 
// FlowSingle
//

pub struct FlowSingle<I: FlowIn<I>, O: FlowIn<O>> {
    sources_outer: SourcesOuter,
    sources_inner: SourcesInner,

    input_ids: I::Nodes,

    output_id: NodeId,
    //output: In<OutputData<O>>,
    output: SharedOutput<O>,
}

impl<I, O> FlowSingle<I, O>
where
    I: FlowIn<I>,
    O: FlowIn<O>
{
    pub fn next(&mut self, waker: &mut Dispatcher) -> Option<O> {
       //  self.output.fill(&mut waker.inner());

        while waker.outer().apply(&mut self.sources_outer) ||
            waker.inner().apply(&mut self.sources_inner) {
        }

        // assert!(self.output.fill(&mut waker.inner()));

        match self.output.take() {
            Some(value) => Some(value),
            None => None,
        }
    }

    fn init(&mut self) -> Dispatcher {
        let dispatcher = Dispatcher::new();

        //self.output.init();
        self.output.take();
        self.sources_outer.init();
        self.sources_inner.init();

        dispatcher
    }
}

impl<I, O> FlowTrait<I, O> for FlowSingle<I, O>
where
    I: FlowIn<I>,
    O: FlowIn<O>
{
    fn call(&mut self, input: I) -> Option<O> {
        let mut iter = self.iter(input);

        iter.next()
    }

    fn iter(&mut self, _input: I) -> FlowIter<I, O> {
        // let mut data = self.new_data();

        // In::write(&self.input, &mut data, input);

        let dispatcher = self.init(); // &mut data);

        FlowIter {
            flow: self,
            // data: data,
            waker: dispatcher,
            count: 0,
        }
    }
}

pub struct FlowIter<'a, I: FlowIn<I>, O: FlowIn<O>> {
    flow: &'a mut FlowSingle<I, O>,
    waker: Dispatcher,
    count: usize,
}

impl<I: FlowIn<I>, O: FlowIn<O>> Iterator for FlowIter<'_, I, O> {
    type Item = O;

    fn next(&mut self) -> Option<Self::Item> {
        self.flow.next(&mut self.waker)
    }
}

//
// Flow threading
//

//
// Flow builder
//

pub struct FlowBuilder<I: FlowIn<I>, O: FlowIn<O>> {
    sources: SourcesBuilder,
    nil_id: SourceId<()>,
    in_nodes: I::Nodes,
    marker: PhantomData<(I, O)>,
}

impl<I: FlowIn<I>, O: FlowIn<O>> FlowBuilder<I, O> {
    fn new() -> Self {
        let mut tasks = SourcesBuilder::new();

        let nil_id = tasks.push_source(Box::new(NilTask), &());

        let input_id = I::new_flow_input(&mut tasks);

        let builder = FlowBuilder {
            sources: tasks,
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

    pub fn nil(&self) -> SourceId<()> {
        self.nil_id.clone()
    }

    pub fn task<I1, O1>(
        &mut self, 
        task: impl Source<I1, O1>,
        in_nodes: &I1::Nodes,
    ) -> SourceId<O1>
    where
        I1: FlowIn<I1>,
        O1: Send + Clone + 'static,
    {
        self.sources.push_source(Box::new(task), in_nodes)
    }

    pub fn output(mut self, src_nodes: &O::Nodes) -> Flow<I, O> {
        let mut output_ids = Vec::new();

        O::node_ids(src_nodes, &mut output_ids);

        let output_task = OutputTask::<O>::new();
        let shared_output = output_task.data().clone();

        let id = self.sources.push_source(Box::new(output_task), src_nodes);
        let tail = self.sources.push_source(Box::new(TailTask), &id);

        //let source = self.sources.add_pipe_nil(id.clone(), tail.id(), 0);
        // self.sources.add_pipe_nil(id.clone(), tail.id(), 0);

        let Sources(outer, inner) = self.sources.build();

        let is_pool = true;

        if is_pool {
            FlowPool::<I, O>::new(
                outer, 
                inner, 
                self.in_nodes, 
                shared_output, // In::new(source), 
                tail.id(),
            )
        } else {
            Flow::new(FlowSingle {
                sources_outer: outer,
                sources_inner: inner,

                input_ids: self.in_nodes,
                output: shared_output, // In::new(source),
                output_id: id.id(),
            })
        }
    }
}

//
// Output task
//
#[derive(Clone)]
pub struct OutputData<T: Clone + Send + 'static> {
    marker: PhantomData<T>,
}

impl<T: Clone + Send + 'static> FlowData for OutputData<T> {}

struct OutputTask<O: FlowIn<O>> {
    data: SharedOutput<O>,
}

impl<O: FlowIn<O>> OutputTask<O> {
    pub(crate) fn new() -> Self {
        Self {
            data: SharedOutput::new(),
        }
    }

    fn data(&self) -> &SharedOutput<O> {
        &self.data
    }
}

impl<O: FlowIn<O>> Source<O, bool> for OutputTask<O> {
    fn next(&mut self, input: &mut O::Input) -> source::Result<Out<bool>> {
        let value = O::next(input);

        match value {
            Out::Some(value) => {
                self.data.replace(value);
                Ok(Out::Some(true))
            },
            Out::None => {
                self.data.take();
                Ok(Out::None)
            },
            Out::Pending => {
                Ok(Out::Pending)
            }
        }
    }
}

#[derive(Clone)]
struct SharedOutput<O> {
    value: Arc<Mutex<Option<O>>>,
}

impl<O> SharedOutput<O> {
    fn new() -> Self {
        Self {
            value: Arc::new(Mutex::new(None)),
        }
    }

    fn take(&self) -> Option<O> {
        self.value.lock().unwrap().take()
    }

    fn replace(&self, value: O) {
        self.value.lock().unwrap().replace(value);
    }
}

//
// tail task
//

struct TailTask;

impl Source<bool, ()> for TailTask {
    fn next(&mut self, source: &mut In<bool>) -> source::Result<Out<()>> {
        source.next(); // assert!(if let Some(true) = source.next() { true } else { false });

        Ok(Out::Some(()))
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
                    Ok(Out::Some(v))
                },
                None => {
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
        assert_eq!(take(&vec), "Node0[], Node1[Some(\"b\")]");

        assert_eq!(flow.call(()), None);
        assert_eq!(take(&vec), "Node0[], Node1[Some(\"a\")]");

        assert_eq!(flow.call(()), None);
        assert_eq!(take(&vec), "Node0[], Node1[None]");

        assert_eq!(flow.call(()), None);
        assert_eq!(take(&vec), "Node0[], Node1[None]");
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
        
        let mut builder = Flow::<(), Vec<usize>>::builder();

        let ptr = vec.clone();
        let n_0 = builder.task::<(), usize>(move |_: &mut ()| {
            ptr.lock().unwrap().push(format!("N-0[]"));
            Ok(Out::Some(1))
        }, &()); // builder.nil());

        let ptr = vec.clone();
        let n_1 = builder.task::<usize, usize>(move |x: &mut In<usize>| {
            ptr.lock().unwrap().push(format!("N-1[{}]", x.next().unwrap()));
            Ok(Out::None)
        }, &n_0);

        let ptr = vec.clone();
        let n_2 = builder.task::<usize, usize>(move |x: &mut In<usize>| {
            ptr.lock().unwrap().push(format!("N-1[{}]", x.next().unwrap()));
            Ok(Out::None)
        }, &n_0);

        let mut flow = builder.output(&(vec![n_1, n_2])); // n_2);

        assert_eq!(flow.call(()), None);
        assert_eq!(take(&vec), "N-0[], N-0[], N-1[1], N-1[1]");
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