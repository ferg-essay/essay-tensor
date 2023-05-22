use super::{
    task::{TasksBuilder, InMeta, NodeId, TaskId}, 
    dispatch::{InnerWaker}, 
    pipe::{Out, In}
};

pub trait FlowData : Send + Clone + 'static {}

pub trait FlowIn<T: Send> : Send + Clone + 'static {
    type Nodes : Clone;
    type Input : Send;

    // execution methods

    fn init_source(source: &mut Self::Input);

    fn fill_source(
        source: &mut Self::Input, 
        infos: &mut Vec<InMeta>, 
        index: &mut usize, 
        waker: &mut dyn InnerWaker
    ) -> bool;

    fn next(source: &mut Self::Input) -> Out<T>;

    //
    // build methods
    //

    fn node_ids(nodes: &Self::Nodes, ids: &mut Vec<NodeId>);

    fn new_flow_input(tasks: &mut TasksBuilder) -> Self::Nodes;

    fn new_input(
        dst_id: NodeId, 
        src_nodes: &Self::Nodes, 
        infos: &mut Vec<InMeta>,
        tasks: &mut TasksBuilder,
    ) -> Self::Input;
}

impl FlowIn<()> for () {
    type Nodes = (); // TaskId<()>;
    type Input = ();

    fn init_source(_source: &mut Self::Input) {
    }

    fn fill_source(
        _source: &mut Self::Input, 
        _infos: &mut Vec<InMeta>, 
        _index: &mut usize, 
        _waker: &mut dyn InnerWaker
    ) -> bool {
        true
    }

    fn next(_source: &mut Self::Input) -> Out<()> {
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

    fn new_flow_input(_tasks: &mut TasksBuilder) -> Self::Nodes {
        /*
        let node = InputTask::<()>::new(id);

        tasks.push_task(Box::new(node))
        */
        ()
    }

    fn new_input(
        _dst_id: NodeId,
        _src_nodes: &Self::Nodes, 
        _infos: &mut Vec<InMeta>,
        _tasks: &mut TasksBuilder,
    ) -> Self::Input {
        ()
    }
}

impl<T:FlowData + Clone + 'static> FlowIn<T> for T {
    type Nodes = TaskId<T>;
    type Input = In<T>;

    fn new_flow_input(tasks: &mut TasksBuilder) -> Self::Nodes {
        /*
        let id = graph.push_input::<T>();

        let node = InputTask::<T>::new(id.clone());

        task_graph.push_node(Box::new(node));

        id
        */
        todo!()
    }

    fn init_source(input: &mut Self::Input) {
        input.init();
    }

    fn fill_source(
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

    fn next(source: &mut Self::Input) -> Out<T> {
        match source.next() {
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
        tasks: &mut TasksBuilder
    ) -> Self::Input {
        let src_id = in_nodes;

        let dst_index = input_meta.len();

        let source = tasks.add_pipe(src_id.clone(), dst_id, dst_index);

        input_meta.push(InMeta::new(src_id.id(), source.out_index()));

        In::new(source)
    }
}

impl<T: FlowIn<T>> FlowIn<Vec<T>> for Vec<T> {
    type Nodes = Vec<T::Nodes>;
    type Input = Vec<T::Input>;

    fn init_source(
        source: &mut Self::Input, 
    ) {
        for source in source {
            T::init_source(source);
        }
    }

    fn fill_source(
        source: &mut Self::Input, 
        infos: &mut Vec<InMeta>, 
        index: &mut usize, 
        waker: &mut dyn InnerWaker
    ) -> bool {
        for source in source {
            if ! T::fill_source(source, infos, index, waker) {
                return false;
            }
        }

        true
    }

    fn next(source: &mut Self::Input) -> Out<Vec<T>> {
        let mut vec = Vec::new();

        for source in source {
            match T::next(source) {
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
        src_nodes: &Self::Nodes, 
        infos: &mut Vec<InMeta>,
        tasks: &mut TasksBuilder
    ) -> Self::Input {
        let mut vec = Vec::new();

        for node in src_nodes.iter() {
            vec.push(T::new_input(dst_id, node, infos, tasks));
        }

        vec
    }

    fn new_flow_input(_tasks: &mut TasksBuilder) -> Self::Nodes {
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
        
            fn new_flow_input(tasks: &mut TasksBuilder) -> Self::Nodes {
                let key = ($(
                    $t::new_flow_input(tasks)
                ),*);

                //let task = InputTask::<($($t),*)>::new(key.clone());

                // tasks.push_node(Box::new(task));

                key
            }

            fn init_source(
                source: &mut Self::Input, 
            ) {
                let ($($v),*) = source;
        
                $(
                    $t::init_source($v);
                )*
            }

            fn fill_source(
                input: &mut Self::Input, 
                meta: &mut Vec<InMeta>, 
                index: &mut usize, 
                waker: &mut dyn InnerWaker
            ) -> bool {
                let ($($v),*) = input;
        
                $(
                    if ! $t::fill_source($v, meta, index, waker) {
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
                nodes: &Self::Nodes,
                ids: &mut Vec<NodeId>, 
            ) {
                let ($($t),*) = nodes;

                $(
                    $t::node_ids($t, ids);
                )*
            }

            fn new_input(
                dst_id: NodeId,
                src_nodes: &Self::Nodes, 
                input_meta: &mut Vec<InMeta>,
                tasks: &mut TasksBuilder
            ) -> Self::Input {
                let ($($t),*) = src_nodes;

                ($(
                    $t::new_input(dst_id, $t, input_meta, tasks)
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
