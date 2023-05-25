use super::{
    source::{SourceId}, 
    data::{FlowIn, FlowData}, SourceFactory,
};

pub trait Flow<I: FlowIn<I>, O: FlowIn<O>> {
    type Iter<'a> : Iterator<Item=O> where Self: 'a;

    fn call(&mut self, input: I) -> Option<O> {
        self.iter(input).next()
    }

    fn iter<'a>(&'a mut self, input: I) -> Self::Iter<'a>;
}

pub trait FlowTrait<I: FlowIn<I>, O: FlowIn<O>> {
    type Iter<'a> : Iterator<Item=O> where Self: 'a;

    fn iter<'a>(&'a mut self, input: I) -> Self::Iter<'a>;
}

/*
pub struct FlowBuilder<In: FlowIn<In>, Out: FlowIn<Out>> {
    // flow: Box<dyn FlowBuilderTrait<In, Out>>, 
    marker: PhantomData<(In, Out)>,
}

impl<In: FlowIn<In>, Out: FlowIn<Out>> FlowBuilder<In, Out> {
    fn source<I, O>(
        &mut self, 
        source: impl Into<Box<dyn SourceFactory<I, O>>>,
        in_nodes: &I::Nodes,
    ) -> SourceId<O>
    where
        I: FlowIn<I>,
        O: FlowData,
    {
        //self.flow.source(source, in_nodes)
        todo!()
    }

    fn output(self, src_nodes: &Out::Nodes) -> FlowSingle<In, Out> {
        todo!()
    }
}
*/

//pub struct FlowBuilder<In: FlowIn<In>> {
//builder: Box<dyn FlowOutputBuilder<Flow<O> = TopFlow<In, O>>>,
//}

// pub trait FlowBuilder<In: FlowIn<In>, Out: FlowIn<Out>> : Sized + 'static {
pub trait FlowSourcesBuilder {
    fn source<I, O>(
        &mut self, 
        source: impl SourceFactory<I, O>,
        in_nodes: &I::Nodes,
    ) -> SourceId<O>
    where
        I: FlowIn<I>,
        O: FlowData;
}
pub trait FlowOutputBuilder<In: FlowIn<In>> : FlowSourcesBuilder + Sized + 'static {
    type Flow<O: FlowIn<O>>;

    fn input(&mut self) -> &In::Nodes;

    fn output<O: FlowIn<O>>(self, src_nodes: &O::Nodes) -> Self::Flow<O>; // FlowSingle<In, Out>;
}
/*
pub struct FlowIterBase<'a, O> {
    flow: Box<dyn Iterator<Item=O>>,
    marker: PhantomData<&'a u8>
}

impl<O: FlowIn<O>> Iterator for FlowIterBase<'_, O> {
    type Item = O;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}
*/
/*
pub trait FlowTrait<I: FlowIn<I>, O: FlowIn<O>> : 'static {
    // type Iter<'a> : Iterator<Item=O>;

    /*
    fn builder() -> FlowBuilderPool<I, O> {
        FlowBuilderPool::new()
    }
    */

    fn call(&mut self, input: I) -> Option<O> {
        self.iter(input).next()
    }

    fn iter<'a>(&'a mut self, input: I) -> FlowIterBase<'a, O>; // FlowIter<I, O>; // FlowIter<I, O>;
    // fn next(&mut self) -> Option<O>;
}
*/

/*
pub struct Flow<I: FlowIn<I>, O: FlowIn<O>> {
    flow: dyn FlowTrait<I, O>,
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
 */

//
// Flow threading
//

#[cfg(test)]
mod test {
    use std::{sync::{Arc, Mutex}};

    use source::Source;

    use crate::flow::{
        pipe::{In}, SourceFactory, FlowIn, flow_pool::{self, PoolFlowBuilder}, 
        flow::{Flow, FlowSourcesBuilder},
        FlowOutputBuilder, source::{self, Out}, FlowData,
    };

    #[test]
    fn test_graph_nil() {
        let builder = PoolFlowBuilder::<()>::new();
        let mut flow = builder.output::<()>(&());

        assert_eq!(flow.call(()), None);
    }

    #[test]
    fn test_graph_node() {
        let vec = Arc::new(Mutex::new(Vec::<String>::default()));
        
        let mut builder = PoolFlowBuilder::<()>::new();
        let ptr = vec.clone();

        let node_id = builder.source::<(), usize>(S(move |_: &mut ()| {
            ptr.lock().unwrap().push(format!("Node[]"));
            Ok(Out::None)
        }), &());

        assert_eq!(node_id.index(), 0);

        let mut flow = builder.output::<usize>(&node_id);

        assert_eq!(flow.call(()), None);
        assert_eq!(take(&vec), "Node[]");

        assert_eq!(flow.call(()), None);
        assert_eq!(take(&vec), "Node[]");
    }

    #[test]
    fn test_graph_detached_node() {
        let vec = Arc::new(Mutex::new(Vec::<String>::default()));
        
        let mut builder = PoolFlowBuilder::<()>::new();
        let ptr = vec.clone();

        let node_id = builder.source::<(), bool>(S(move |_: &mut ()| {
            ptr.lock().unwrap().push(format!("Node[]"));
            Ok(Out::None)
        }), &());

        assert_eq!(node_id.index(), 1);

        // let nil = builder.nil();
        let mut flow = builder.output::<()>(&());

        assert_eq!(flow.call(()), None);
        assert_eq!(take(&vec), "");

        assert_eq!(flow.call(()), None);
        assert_eq!(take(&vec), "");
    }

    #[test]
    fn graph_sequence() {
        let vec = Arc::new(Mutex::new(Vec::<String>::default()));
        
        let mut builder = PoolFlowBuilder::<()>::new();

        let ptr = vec.clone();
        let mut data = vec!["a".to_string(), "b".to_string()];

        let n_0 = builder.source::<(), String>(S(move |_: &mut ()| {
            ptr.lock().unwrap().push(format!("Node0[]"));
            match data.pop() {
                Some(v) => {
                    Ok(Out::Some(v))
                },
                None => {
                    Ok(Out::None)
                }
            }
        }), &());

        assert_eq!(n_0.index(), 1);

        let ptr = vec.clone();
        let n_1 = builder.source::<String, bool>(S(move |s: &mut In<String>| {
            ptr.lock().unwrap().push(format!("Node1[{:?}]", s.next()));
            Ok(Out::None)
        }), &n_0);

        assert_eq!(n_1.index(), 2);

        let mut flow = builder.output::<bool>(&n_1); // n_1);

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
        
        let mut builder = PoolFlowBuilder::<usize>::new();

        let ptr = vec.clone();

        let input = builder.input().clone();
        let _n_0 = builder.source::<usize, bool>(S(move |x: &mut In<usize>| {
            ptr.lock().unwrap().push(format!("Task[{:?}]", x.next().unwrap()));
            Ok(Out::None)
        }), &input);

        let mut flow = builder.output::<()>(&()); // n_0);

        assert_eq!(flow.call(1), Some(()));
        assert_eq!(take(&vec), "Task[1]");

        assert_eq!(flow.call(2), Some(()));
        assert_eq!(take(&vec), "Task[2]");
    }

    #[test]
    fn flow_output() {
        let vec = Arc::new(Mutex::new(Vec::<String>::default()));
        
        let mut builder = PoolFlowBuilder::<()>::new();

        let ptr = vec.clone();

        let mut count = 2;
        let n_0 = builder.source::<(), usize>(S(move |_: &mut ()| {
            ptr.lock().unwrap().push(format!("Task[{}]", count));
            if count > 0 {
                count -= 1;
                Ok(Out::Some(count))
            } else {
                Ok(Out::None)
            }
        }), &());

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
        
        let mut builder = PoolFlowBuilder::<usize>::new();

        let ptr = vec.clone();

        let input = builder.input().clone();
        let n_0 = builder.source::<usize, usize>(S(move |x: &mut In<usize>| {
            let x_v = x.next().unwrap();
            ptr.lock().unwrap().push(format!("Task[{:?}]", x_v));
            Ok(Out::Some(x_v + 10))
        }), &input);

        let mut flow = builder.output(&n_0);

        assert_eq!(flow.call(1), Some(11));
        assert_eq!(take(&vec), "Task[1]");

        assert_eq!(flow.call(2), Some(12));
        assert_eq!(take(&vec), "Task[2]");
    }

    #[test]
    fn node_output_split() {
        let vec = Arc::new(Mutex::new(Vec::<String>::default()));
        
        let mut builder = PoolFlowBuilder::<()>::new();

        let ptr = vec.clone();
        let n_0 = builder.source::<(), usize>(S(move |_: &mut ()| {
            ptr.lock().unwrap().push(format!("N-0[]"));
            Ok(Out::Some(1))
        }), &()); // builder.nil());

        let ptr = vec.clone();
        let n_1 = builder.source::<usize, usize>(S(move |x: &mut In<usize>| {
            ptr.lock().unwrap().push(format!("N-1[{}]", x.next().unwrap()));
            Ok(Out::None)
        }), &n_0);

        let ptr = vec.clone();
        let n_2 = builder.source::<usize, usize>(S(move |x: &mut In<usize>| {
            ptr.lock().unwrap().push(format!("N-1[{}]", x.next().unwrap()));
            Ok(Out::None)
        }), &n_0);

        let mut flow = builder.output::<Vec<usize>>(&(vec![n_1, n_2])); // n_2);

        assert_eq!(flow.call(()), None);
        assert_eq!(take(&vec), "N-0[], N-0[], N-1[1], N-1[1]");
    }

    #[test]
    fn node_tuple_input() {
        let vec = Arc::new(Mutex::new(Vec::<String>::default()));
        
        let mut builder = PoolFlowBuilder::<()>::new();

        let ptr = vec.clone();
        let n_1 = builder.source::<(), usize>(S(move |_: &mut ()| {
            ptr.lock().unwrap().push(format!("N-1[]"));
            Ok(Out::Some(1))
        }), &()); // builder.nil());

        let ptr = vec.clone();
        let n_2 = builder.source::<(), f32>(S(move |_: &mut ()| {
            ptr.lock().unwrap().push(format!("N-1[]"));
            Ok(Out::Some(10.5))
        }), &()); // builder.nil());

        let ptr = vec.clone();
        let _n_2 = builder.source::<(usize, f32), bool>(S(move |v: &mut (In<usize>, In<f32>)| {
            ptr.lock().unwrap().push(format!("N-2[{}, {}]", v.0.next().unwrap(), v.1.next().unwrap()));
            Ok(Out::None)
        }), &(n_1, n_2));

        let mut flow = builder.output(&()); // n_2);

        assert_eq!(flow.call(()), Some(()));
        assert_eq!(take(&vec), "N-1[], N-1[], N-2[1, 10.5]");
    }

    #[test]
    fn node_vec_input() {
        let vec = Arc::new(Mutex::new(Vec::<String>::default()));
        
        let mut builder = PoolFlowBuilder::<()>::new();

        let ptr = vec.clone();
        let n_1 = builder.source::<(), usize>(S(move |_: &mut ()| {
            ptr.lock().unwrap().push(format!("N-1[]"));
            Ok(Out::Some(1))
        }), &()); // builder.nil());

        let ptr = vec.clone();
        let n_2 = builder.source::<(), usize>(S(move |_: &mut ()| {
            ptr.lock().unwrap().push(format!("N-1[]"));
            Ok(Out::Some(10))
        }), &()); // builder.nil());

        let ptr = vec.clone();
        let n_3 = builder.source::<(), usize>(S(move |_: &mut ()| {
            ptr.lock().unwrap().push(format!("N-1[]"));
            Ok(Out::Some(100))
        }), &()); // builder.nil());

        let ptr = vec.clone();
        let _n_4 = builder.source::<Vec<usize>, bool>(S(move |x: &mut Vec<In<usize>>| {
            ptr.lock().unwrap().push(format!("N-2[{:?}]", x[0].next().unwrap()));
            Ok(Out::None)
        }), &vec![n_1, n_2, n_3]);

        let mut flow = builder.output(&()); // n_4);

        assert_eq!(flow.call(()), Some(()));
        assert_eq!(take(&vec), "N-1[], N-1[], N-1[], N-2[[1, 10, 100]]");
    }

    #[test]
    fn output_with_incomplete_data() {
        todo!();
        /*
        let vec = Arc::new(Mutex::new(Vec::<String>::default()));
        
        let mut builder = PoolFlowBuilder::<usize>::new();

        let ptr = vec.clone();

        let input = builder.input();
        let n_0 = builder.source::<usize, usize>(move |x: &mut In<usize>| {
            ptr.lock().unwrap().push(format!("Task[{:?}]", x.next().unwrap()));
            Ok(Out::None)
        }, &input);

        let mut flow = builder.output::<usize>(&n_0);

        assert_eq!(flow.call(1), None);
        assert_eq!(take(&vec), "Task[1]");

        assert_eq!(flow.call(2), None);
        assert_eq!(take(&vec), "Task[2]");
        */
    }

    #[test]
    fn graph_iter() {
        let vec = Arc::new(Mutex::new(Vec::<String>::default()));
        
        let mut builder = PoolFlowBuilder::<()>::new();
        let ptr = vec.clone();

        let n_0 = builder.source::<(), usize>(S(move |_: &mut ()| {
            ptr.lock().unwrap().push(format!("Node[]"));
            Ok(Out::Some(1))
        }), &()); // builder.nil());

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
    /*
    impl<I, O, F> From<F> for Box<dyn SourceFactory<I, O>>
    where
        I: FlowIn<I> + 'static,
        O: FlowData,
        F: FnMut(&mut I::Input) -> source::Result<Out<O>> + Send + 'static
    {
        fn from(value: F) -> Self {
            let mut item = Some(Box::new(value));
            Box::new(move || item.take().unwrap())
        }
    }
    */
    /*
    impl<I, O, F> SourceFactory<I, O> for F
    where
        I: FlowIn<I> + 'static,
        O: FlowData,
        F: FnMut(&mut I::Input) -> source::Result<Out<O>> + Send + 'static
    {
        fn new(&mut self) -> Box<dyn source::Source<I, O>> {
            Box::new(self.clone());    
        }
    }
    */

    struct Wrap<I: FlowIn<I>, O: FlowIn<O>> {
        source: Option<Box<dyn Source<I, O>>>,
    }

    fn S<I: FlowIn<I>, O: FlowIn<O>>(source: impl Source<I, O>) -> Wrap<I, O> {
        Wrap::new(source)
    }

    impl<I: FlowIn<I>, O: FlowIn<O>> Wrap<I, O> {
        fn new(source: impl Source<I, O>) -> Self {
            Self {
                source: Some(Box::new(source)),
            }
        }
    }

    impl<I: FlowIn<I>, O: FlowIn<O>> SourceFactory<I, O> for Wrap<I, O> {
        fn new(&mut self) -> Box<dyn Source<I, O>> {
            self.source.take().unwrap()
        }
    }
}