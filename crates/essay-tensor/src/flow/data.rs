use core::fmt;

use super::{
    flow_pool::{InMeta}, 
    dispatch::{InnerWaker}, 
    pipe::{In, PipeIn}, source::NodeId, SourceId, Out, SourceFactory
};

pub trait FlowData : Send + Sync + Clone + fmt::Debug + 'static {}

pub trait FlowIn<T: Send> : Send + Clone + 'static {
    type Nodes : Clone;
    type Input : Send;

    // execution methods

    fn init(input: &mut Self::Input);

    fn fill(
        input: &mut Self::Input, 
        input_meta: &mut Vec<InMeta>, 
        index: &mut usize, 
        waker: &mut dyn InnerWaker
    ) -> bool;

    fn next(input: &mut Self::Input) -> Out<T>;

    //
    // build methods
    //

    fn node_ids(nodes: &Self::Nodes, ids: &mut Vec<NodeId>);

    fn new_flow_input(builder: &mut impl FlowInBuilder) -> Self::Nodes;

    fn new_input(
        dst_id: NodeId, 
        src_nodes: &Self::Nodes, 
        input_meta: &mut Vec<InMeta>,
        builder: &mut impl FlowInBuilder,
    ) -> Self::Input;
    
}
pub trait FlowInBuilder {
    fn add_source<I, O>(
        &mut self, 
        source: impl SourceFactory<I, O>,
        in_nodes: &I::Nodes,
    ) -> SourceId<O>
    where
        I: FlowIn<I>,
        O: FlowData;

    fn add_pipe<O: FlowIn<O>>(
        &mut self,
        src_id: SourceId<O>,
        dst_id: NodeId,
        dst_index: usize,
    ) -> Box<dyn PipeIn<O>>;
}

impl FlowIn<()> for () {
    type Nodes = (); // TaskId<()>;
    type Input = ();

    fn init(_input: &mut Self::Input) {
    }

    fn fill(
        _input: &mut Self::Input, 
        _input_meta: &mut Vec<InMeta>, 
        _index: &mut usize, 
        _waker: &mut dyn InnerWaker
    ) -> bool {
        true
    }

    fn next(_input: &mut Self::Input) -> Out<()> {
        Out::None
    }

    //
    // builder
    //

    fn node_ids(
        _nodes: &Self::Nodes,
        _ids: &mut Vec<NodeId>, 
    ) {
        // ids.push(nodes.id());
    }

    fn new_flow_input(_builder: &mut impl FlowInBuilder) -> Self::Nodes {
        /*
        let node = InputTask::<()>::new(id);

        tasks.push_task(Box::new(node))
        */
        ()
    }

    fn new_input(
        _dst_id: NodeId,
        _in_nodes: &Self::Nodes, 
        _input_meta: &mut Vec<InMeta>,
        _builder: &mut impl FlowInBuilder,
    ) -> Self::Input {
        ()
    }
}

impl<T:FlowData> FlowIn<T> for T {
    type Nodes = SourceId<T>;
    type Input = In<T>;

    fn init(input: &mut Self::Input) {
        input.init();
    }

    fn fill(
        input: &mut Self::Input, 
        input_meta: &mut Vec<InMeta>, 
        index: &mut usize, 
        waker: &mut dyn InnerWaker
    ) -> bool {
        let is_available = input.fill(waker);

        input_meta[*index].set_n_read(input.n_read());

        *index += 1;

        is_available
    }

    fn next(input: &mut Self::Input) -> Out<T> {
        match input.next() {
            Some(value) => Out::Some(value),
            None => Out::None,
        }
    }

    //
    // builder
    //

    fn node_ids(
        nodes: &Self::Nodes,
        ids: &mut Vec<NodeId>, 
    ) {
        ids.push(nodes.id());
    }

    fn new_input(
        dst_id: NodeId,
        in_nodes: &Self::Nodes, 
        input_meta: &mut Vec<InMeta>,
        builder: &mut impl FlowInBuilder
    ) -> Self::Input {
        let src_id = in_nodes;

        let dst_index = input_meta.len();

        let input = builder.add_pipe(src_id.clone(), dst_id, dst_index);
        input_meta.push(InMeta::new(src_id.id(), input.out_index()));

        In::new(input)
    }

    fn new_flow_input(_builder: &mut impl FlowInBuilder) -> Self::Nodes {
        /*
        let id = graph.push_input::<T>();

        let node = InputTask::<T>::new(id.clone());

        task_graph.push_node(Box::new(node));

        id
        */
        todo!()
    }
}

impl<T: FlowIn<T>> FlowIn<Vec<T>> for Vec<T> {
    type Nodes = Vec<T::Nodes>;
    type Input = Vec<T::Input>;

    fn init(
        input: &mut Self::Input, 
    ) {
        for input in input {
            T::init(input);
        }
    }

    fn fill(
        input: &mut Self::Input, 
        input_meta: &mut Vec<InMeta>, 
        index: &mut usize, 
        waker: &mut dyn InnerWaker
    ) -> bool {
        for input in input {
            if ! T::fill(input, input_meta, index, waker) {
                return false;
            }
        }

        true
    }

    fn next(input: &mut Self::Input) -> Out<Vec<T>> {
        let mut vec = Vec::new();

        for input in input {
            match T::next(input) {
                Out::None => return Out::None,
                Out::Some(value) => vec.push(value),
                Out::Pending => todo!(),
            }
        }

        Out::Some(vec)
    }

    //
    // builders
    //

    fn node_ids(
        nodes: &Self::Nodes,
        ids: &mut Vec<NodeId>, 
    ) {
        for node_in in nodes {
            T::node_ids(node_in, ids);
        }
    }

    fn new_input(
        dst_id: NodeId,
        in_nodes: &Self::Nodes, 
        input_meta: &mut Vec<InMeta>,
        builder: &mut impl FlowInBuilder
    ) -> Self::Input {
        let mut vec = Vec::new();

        for node in in_nodes.iter() {
            vec.push(T::new_input(dst_id, node, input_meta, builder));
        }

        vec
    }

    fn new_flow_input(_tasks: &mut impl FlowInBuilder) -> Self::Nodes {
        todo!();
    }
}

macro_rules! tuple_flow {
    ($(($t:ident, $v:ident)),*) => {
        #[allow(non_snake_case)]
        impl<$($t),*> FlowIn<($($t),*)> for ($($t),*)
        where $(
            $t: FlowIn<$t>,
        )*
        {
            type Nodes = ($($t::Nodes),*);
            type Input = ($($t::Input),*);
        
            fn new_flow_input(builder: &mut impl FlowInBuilder) -> Self::Nodes {
                let key = ($(
                    $t::new_flow_input(builder)
                ),*);

                //let task = InputTask::<($($t),*)>::new(key.clone());

                // tasks.push_node(Box::new(task));

                key
            }

            fn init(
                input: &mut Self::Input, 
            ) {
                let ($($v),*) = input;
        
                $(
                    $t::init($v);
                )*
            }

            fn fill(
                input: &mut Self::Input, 
                input_meta: &mut Vec<InMeta>, 
                index: &mut usize, 
                waker: &mut dyn InnerWaker
            ) -> bool {
                let ($($v),*) = input;
        
                $(
                    if ! $t::fill($v, input_meta, index, waker) {
                        return false
                    }
                )*

                true
            }

            fn next(input: &mut Self::Input) -> Out<($($t),*)> {
                let ($($v),*) = input;
                
                let value = ($(
                    match $t::next($v) {
                        Out::None => return Out::None,
                        Out::Some(value) => value,
                        Out::Pending => todo!(),
                    }
                ),*);
        
                Out::Some(value)
            }
        
            //
            // builders
            //

            fn node_ids(
                in_nodes: &Self::Nodes,
                ids: &mut Vec<NodeId>, 
            ) {
                let ($($t),*) = in_nodes;

                $(
                    $t::node_ids($t, ids);
                )*
            }

            fn new_input(
                dst_id: NodeId,
                in_nodes: &Self::Nodes, 
                input_meta: &mut Vec<InMeta>,
                builder: &mut impl FlowInBuilder
            ) -> Self::Input {
                let ($($t),*) = in_nodes;

                ($(
                    $t::new_input(dst_id, $t, input_meta, builder)
                ),*)
            }
        }
    }
}

tuple_flow!((T1, v1), (T2, v2));
tuple_flow!((T1, v1), (T2, v2), (T3, v3));
tuple_flow!((T1, v1), (T2, v2), (T3, v3), (T4, v4));
tuple_flow!((T1, v1), (T2, v2), (T3, v3), (T4, v4), (T5, v5));
tuple_flow!((T1, v1), (T2, v2), (T3, v3), (T4, v4), (T5, v5), (T6, v6));
tuple_flow!((T1, v1), (T2, v2), (T3, v3), (T4, v4), (T5, v5), (T6, v6), (T7, v7));
tuple_flow!((T1, v1), (T2, v2), (T3, v3), (T4, v4), (T5, v5), (T6, v6), (T7, v7), (T8, v8));

// impl FlowData for () {}
impl FlowData for String {}
impl FlowData for usize {}
impl FlowData for i32 {}
impl FlowData for f32 {}
impl FlowData for bool {}
