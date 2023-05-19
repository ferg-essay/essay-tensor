use super::{
    task::{Tasks, TasksBuilder, SourceInfo}, 
    dispatch::Dispatcher, 
    graph::{NodeId, TaskId, Graph}, source::{Out, Source}
};

pub trait FlowData : Send + Clone + 'static {}

pub trait FlowIn<T> : Send + Clone + 'static {
    type Nodes : Clone;
    type Source;

    // execution methods

    fn wake(
        nodes: &Self::Nodes, 
        graph: &Graph,
        tasks: &mut Tasks,
        dispatcher: &mut Dispatcher,
        // data: &mut GraphData,
    ) -> Out<()>;

    fn fill_source(source: &mut Self::Source, waker: &mut Dispatcher) -> bool;

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

    fn new_input(graph: &mut Graph, tasks: &mut TasksBuilder) -> Self::Nodes;

    fn new_source(
        dst_id: NodeId, 
        src_nodes: &Self::Nodes, 
        infos: &mut Vec<SourceInfo>,
        graph: &mut Graph,
        tasks: &mut TasksBuilder,
    ) -> Self::Source;
}

impl FlowIn<()> for () {
    type Nodes = (); // TaskId<()>;
    type Source = ();

    fn fill_source(_source: &mut Self::Source, _waker: &mut Dispatcher) -> bool {
        true
    }

    fn new_input(graph: &mut Graph, tasks: &mut TasksBuilder) -> Self::Nodes {
        /*
        let node = InputTask::<()>::new(id);

        tasks.push_task(Box::new(node))
        */
        ()
    }

    fn wake(
        _nodes: &Self::Nodes, 
        _graph: &Graph,
        _tasks: &mut Tasks,
        _dispatcher: &mut Dispatcher,
        // _data: &mut GraphData,
    ) -> Out<()> {
        Out::Some(())
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

    fn new_source(
        dst_id: NodeId,
        src_nodes: &Self::Nodes, 
        infos: &mut Vec<SourceInfo>,
        graph: &mut Graph,
        tasks: &mut TasksBuilder,
    ) -> Self::Source {
        ()
    }
}

impl<T:FlowData + Clone + 'static> FlowIn<T> for T {
    type Nodes = TaskId<T>;
    type Source = Source<T>;

    fn new_input(graph: &mut Graph, task_graph: &mut TasksBuilder) -> Self::Nodes {
        /*
        let id = graph.push_input::<T>();

        let node = InputTask::<T>::new(id.clone());

        task_graph.push_node(Box::new(node));

        id
        */
        todo!()
    }

    /*
    fn new_source(_nodes: &Self::Nodes) -> Self::Source {
        todo!()
        // SourceImpl::<T>::new()
    }
    */

    fn wake(
        nodes: &Self::Nodes, 
        graph: &Graph,
        tasks: &mut Tasks,
        dispatcher: &mut Dispatcher,
        // data: &mut GraphData,
    ) -> Out<()> {
        graph.wake(&nodes.id(), tasks, dispatcher)
    }

    fn fill_source(
        source: &mut Self::Source, 
        waker: &mut Dispatcher,
    ) -> bool {
        source.fill(waker)
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
        graph: &mut Graph,
        tasks: &mut TasksBuilder
    ) -> Self::Source {
        let src_id = src_nodes;

        let dst_index = infos.len();

        graph.add_arrow_out(src_id.id(), dst_id);

        let source = tasks.add_sink(src_id.clone(), dst_id, dst_index);

        infos.push(SourceInfo::new(src_id.id(), source.src_index()));

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

    fn fill_source(
        source: &mut Self::Source, 
        waker: &mut Dispatcher,
    ) -> bool {
        todo!()
    }

    fn wake(
        nodes: &Self::Nodes, 
        graph: &Graph,
        tasks: &mut Tasks,
        dispatcher: &mut Dispatcher,
        // data: &mut GraphData,
    ) -> Out<()> {
        let mut out = Out::Some(());

        for node in nodes {
            match T::wake(node, graph, tasks, dispatcher) {
                Out::None => return Out::None,
                Out::Pending => out = Out::Pending,
                Out::Some(_) => {},
            }
        }

        out
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
        graph: &mut Graph,
        tasks: &mut TasksBuilder
    ) -> Self::Source {
        let mut vec = Vec::new();

        for node in src_nodes.iter() {
            vec.push(T::new_source(dst_id, node, infos, graph, tasks));
        }

        vec
    }

    fn new_input(_graph: &mut Graph, _tasks: &mut TasksBuilder) -> Self::Nodes {
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
        
            fn new_input(graph: &mut Graph, tasks: &mut TasksBuilder) -> Self::Nodes {
                let key = ($(
                    $t::new_input(graph, tasks)
                ),*);

                //let task = InputTask::<($($t),*)>::new(key.clone());

                // tasks.push_node(Box::new(task));

                key
            }

            fn wake(
                nodes: &Self::Nodes, 
                graph: &Graph,
                tasks: &mut Tasks,
                dispatcher: &mut Dispatcher,
                // data: &mut GraphData,
            ) -> Out<()> {
                let ($($t),*) = nodes;

                let mut out = Out::Some(());
        
                $(
                    match $t::wake($t, graph, tasks, dispatcher) {
                        Out::None => return Out::None,
                        Out::Pending => out = Out::Pending,
                        Out::Some(_) => {},
                    };
                )*
        
                out
            }

            fn fill_source(
                source: &mut Self::Source, 
                waker: &mut Dispatcher,
            ) -> bool {
                let ($($v),*) = source;
        
                $(
                    if ! $t::fill_source($v, waker) {
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
                graph: &mut Graph,
                tasks: &mut TasksBuilder
            ) -> Self::Source {
                let ($($t),*) = src_nodes;

                ($(
                    $t::new_source(dst_id, $t, infos, graph, tasks)
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
