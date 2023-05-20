use super::{
    task::{TasksBuilder, SourceInfo, NodeId, TaskId, SourceInfos}, 
    dispatch::Dispatcher, 
    source::{Out, Source}
};

pub trait FlowData : Send + Clone + 'static {}

pub trait FlowIn<T> : Send + Clone + 'static {
    type Nodes : Clone;
    type Source;

    // execution methods

    fn init_source(source: &mut Self::Source);

    fn fill_source(
        source: &mut Self::Source, 
        infos: &mut Vec<SourceInfo>, 
        index: &mut usize, 
        waker: &mut Dispatcher
    ) -> bool;

    fn next(source: &mut Self::Source) -> Out<T>;

    //
    // build methods
    //

    fn add_arrows(
        id: NodeId, 
        nodes_in: Self::Nodes, 
        arrows_in: &mut Vec<NodeId>,
        // graph: &mut Graph
    );

    fn node_ids(nodes: &Self::Nodes, ids: &mut Vec<NodeId>);

    fn new_input(tasks: &mut TasksBuilder) -> Self::Nodes;

    fn new_source(
        dst_id: NodeId, 
        src_nodes: &Self::Nodes, 
        infos: &mut Vec<SourceInfo>,
        tasks: &mut TasksBuilder,
    ) -> Self::Source;
}

impl FlowIn<()> for () {
    type Nodes = (); // TaskId<()>;
    type Source = ();

    fn init_source(_source: &mut Self::Source) {
    }

    fn fill_source(
        source: &mut Self::Source, 
        infos: &mut Vec<SourceInfo>, 
        index: &mut usize, 
        waker: &mut Dispatcher
    ) -> bool {
        true
    }

    fn next(source: &mut Self::Source) -> Out<()> {
        Out::None
    }

    //
    // builder
    //

    fn add_arrows(
        id: NodeId,
        node_in: Self::Nodes, 
        arrows_in: &mut Vec<NodeId>, 
    ) {
    }

    fn node_ids(
        nodes: &Self::Nodes,
        ids: &mut Vec<NodeId>, 
    ) {
        // ids.push(nodes.id());
    }

    fn new_input(tasks: &mut TasksBuilder) -> Self::Nodes {
        /*
        let node = InputTask::<()>::new(id);

        tasks.push_task(Box::new(node))
        */
        ()
    }

    fn new_source(
        dst_id: NodeId,
        src_nodes: &Self::Nodes, 
        infos: &mut Vec<SourceInfo>,
        tasks: &mut TasksBuilder,
    ) -> Self::Source {
        ()
    }
}

impl<T:FlowData + Clone + 'static> FlowIn<T> for T {
    type Nodes = TaskId<T>;
    type Source = Source<T>;

    fn new_input(tasks: &mut TasksBuilder) -> Self::Nodes {
        /*
        let id = graph.push_input::<T>();

        let node = InputTask::<T>::new(id.clone());

        task_graph.push_node(Box::new(node));

        id
        */
        todo!()
    }

    fn init_source(source: &mut Self::Source) {
        source.init();
    }

    fn fill_source(
        source: &mut Self::Source, 
        infos: &mut Vec<SourceInfo>, 
        index: &mut usize, 
        waker: &mut Dispatcher
    ) -> bool {
        let is_available = source.fill(waker);

        infos[*index].set_n_read(source.n_read());

        *index += 1;

        is_available
    }

    fn next(source: &mut Self::Source) -> Out<T> {
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

    fn new_source(
        dst_id: NodeId,
        src_nodes: &Self::Nodes, 
        infos: &mut Vec<SourceInfo>,
        tasks: &mut TasksBuilder
    ) -> Self::Source {
        let src_id = src_nodes;

        let dst_index = infos.len();

        let source = tasks.add_sink(src_id.clone(), dst_id, dst_index);

        infos.push(SourceInfo::new(src_id.id(), source.sink_index()));

        Source::new(source)
    }

    fn add_arrows(
        id: NodeId,
        node_in: Self::Nodes, 
        arrows_in: &mut Vec<NodeId>, 
        // graph: &mut Graph
    ) {
        arrows_in.push(node_in.id());

        // graph.add_arrow_out(node_in.id(), id);
    }
}

impl<T: FlowIn<T>> FlowIn<Vec<T>> for Vec<T> {
    type Nodes = Vec<T::Nodes>;
    type Source = Vec<T::Source>;

    fn init_source(
        source: &mut Self::Source, 
    ) {
        for source in source {
            T::init_source(source);
        }
    }

    fn fill_source(
        source: &mut Self::Source, 
        infos: &mut Vec<SourceInfo>, 
        index: &mut usize, 
        waker: &mut Dispatcher
    ) -> bool {
        todo!()
    }

    fn next(source: &mut Self::Source) -> Out<Vec<T>> {
        let mut vec = Vec::new();
        let is_none = true;

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

    fn add_arrows(
        id: NodeId,
        in_nodes: Self::Nodes, 
        in_arrows: &mut Vec<NodeId>, 
        // graph: &mut Graph
    ) {
        for node_in in in_nodes {
            T::add_arrows(id, node_in, in_arrows);
        }
    }

    fn new_source(
        dst_id: NodeId,
        src_nodes: &Self::Nodes, 
        infos: &mut Vec<SourceInfo>,
        tasks: &mut TasksBuilder
    ) -> Self::Source {
        let mut vec = Vec::new();

        for node in src_nodes.iter() {
            vec.push(T::new_source(dst_id, node, infos, tasks));
        }

        vec
    }

    fn new_input(_tasks: &mut TasksBuilder) -> Self::Nodes {
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
            type Source = ($($t::Source),*);
        
            fn new_input(tasks: &mut TasksBuilder) -> Self::Nodes {
                let key = ($(
                    $t::new_input(tasks)
                ),*);

                //let task = InputTask::<($($t),*)>::new(key.clone());

                // tasks.push_node(Box::new(task));

                key
            }

            fn init_source(
                source: &mut Self::Source, 
            ) {
                let ($($v),*) = source;
        
                $(
                    $t::init_source($v);
                )*
            }

            fn fill_source(
                source: &mut Self::Source, 
                infos: &mut Vec<SourceInfo>, 
                index: &mut usize, 
                waker: &mut Dispatcher
            ) -> bool {
                let ($($v),*) = source;
        
                $(
                    if ! $t::fill_source($v, infos, index, waker) {
                        return false
                    }
                )*

                true
            }

            fn next(source: &mut Self::Source) -> Out<($($t),*)> {
                let ($($v),*) = source;
                
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
        
            fn add_arrows(
                id: NodeId,
                nodes_in: Self::Nodes, 
                arrows_in: &mut Vec<NodeId>, 
                // graph: &mut Graph
            ) {
                let ($($t),*) = nodes_in;

                $(
                    $t::add_arrows(id, $t, arrows_in);
                )*
            }

            fn new_source(
                dst_id: NodeId,
                src_nodes: &Self::Nodes, 
                infos: &mut Vec<SourceInfo>,
                tasks: &mut TasksBuilder
            ) -> Self::Source {
                let ($($t),*) = src_nodes;

                ($(
                    $t::new_source(dst_id, $t, infos, tasks)
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
