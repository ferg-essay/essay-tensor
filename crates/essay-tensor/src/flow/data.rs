use std::{any::TypeId, mem::{self, ManuallyDrop}, ptr::NonNull, alloc::Layout};

use super::{
    task::{Tasks, TasksBuilder}, 
    dispatch::Dispatcher, 
    graph::{NodeId, TaskId, Graph}, source::{Out, SourceTrait, Source}
};

pub trait FlowData {}

pub trait FlowIn<T> : Clone + 'static {
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

    fn fill_source(source: &mut Self::Source) -> bool;

    //fn is_available(nodes: &Self::Nodes, data: &GraphData) -> bool;
    //fn read(nodes: &Self::Nodes, data: &mut GraphData) -> T;
    //fn fill_input(source: &mut Self::Source, nodes: &Self::Nodes, data: &mut GraphData) -> bool;

    //fn write(nodes: &Self::Nodes, data: &mut GraphData, value: T) -> bool;

    // build methods

    fn add_arrows(
        id: NodeId, 
        nodes_in: Self::Nodes, 
        arrows_in: &mut Vec<NodeId>,
        // graph: &mut Graph
    );

    fn node_ids(nodes: &Self::Nodes, ids: &mut Vec<NodeId>);

    fn new_input(graph: &mut Graph, tasks: &mut TasksBuilder) -> Self::Nodes;
    // fn new_source(nodes: &Self::Nodes) -> Self::Source;

    fn add_source(
        dst_id: NodeId, 
        src_nodes: &Self::Nodes, 
        graph: &mut Graph,
        tasks: &mut TasksBuilder,
    ) -> Self::Source;
}
/*
pub struct GraphData {
    tasks: Vec<RawData>,
}

impl GraphData {
    pub fn new() -> Self {
        Self {
            tasks: Vec::new(),
        }
    }

    pub fn push<T: Clone + 'static>(&mut self, n_arrows: usize) {
        self.tasks.push(RawData::new::<T>(n_arrows));
    }

    pub fn is_available<T: 'static>(&self, task: &TaskId<T>) -> bool {
        self.tasks[task.index()].is_available()
    }

    pub fn read<T: Clone + 'static>(&mut self, task: &TaskId<T>) -> T {
        self.tasks[task.index()].read()
    }
    
    pub fn write<T: 'static>(&mut self, task: &TaskId<T>, data: T) -> bool {
        self.tasks[task.index()].write(data)
    }
}
*/

impl FlowIn<()> for () {
    type Nodes = (); // TaskId<()>;
    type Source = ();

    /*
    fn new_source(_nodes: &Self::Nodes) -> Self::Source {
        ()
    }
    */

    /*
    fn is_available(_nodes: &Self::Nodes, _data: &GraphData) -> bool {
        true
    }

    fn read(_nodes: &Self::Nodes, _data: &mut GraphData) -> () {
        ()
    }

    fn fill_input(_source: &mut Self::Source, _nodes: &Self::Nodes, _data: &mut GraphData) -> bool {
        true
    }

    fn write(_nodes: &Self::Nodes, _data: &mut GraphData, _value: ()) -> bool {
        false
    }
    */

    fn fill_source(_source: &mut Self::Source) -> bool {
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

    fn add_arrows(
        id: NodeId,
        node_in: Self::Nodes, 
        arrows_in: &mut Vec<NodeId>, 
        // graph: &mut Graph
    ) {
        // arrows_in.push(node_in.id());

        // graph.add_arrow_out(node_in.id(), id);
    }

    fn node_ids(
        nodes: &Self::Nodes,
        ids: &mut Vec<NodeId>, 
    ) {
        // ids.push(nodes.id());
    }

    fn add_source(
        dst_id: NodeId,
        src_nodes: &Self::Nodes, 
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

    /*
    fn is_available(nodes: &Self::Nodes, data: &GraphData) -> bool {
        data.is_available(nodes)
    }

    fn read(nodes: &Self::Nodes, data: &mut GraphData) -> Self {
        data.read(nodes)
    }
    */

    fn fill_source(
        source: &mut Self::Source, 

    ) -> bool {
        todo!()
    }

    /*
    fn write(nodes: &Self::Nodes, data: &mut GraphData, value: Self) -> bool {
        data.write(nodes, value)
    }
    */

    fn node_ids(
        nodes: &Self::Nodes,
        ids: &mut Vec<NodeId>, 
    ) {
        ids.push(nodes.id());
    }

    fn add_source(
        dst_id: NodeId,
        src_nodes: &Self::Nodes, 
        graph: &mut Graph,
        tasks: &mut TasksBuilder
    ) -> Self::Source {
        let src_id = src_nodes;

        graph.add_arrow_out(src_id.id(), dst_id);

        Source::new(tasks.add_sink(src_id.clone(), dst_id))
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
    
    /*
    fn new_source(nodes: &Self::Nodes) -> Self::Source {
        let mut vec = Vec::<T::Source>::new();

        for node in nodes {
            vec.push(T::new_source(node))
        }

        vec
    }
    */

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

    fn add_source(
        dst_id: NodeId,
        src_nodes: &Self::Nodes, 
        graph: &mut Graph,
        tasks: &mut TasksBuilder
    ) -> Self::Source {
        let mut vec = Vec::new();

        for node in src_nodes.iter() {
            vec.push(T::add_source(dst_id, node, graph, tasks));
        }

        vec
    }

    fn new_input(_graph: &mut Graph, _tasks: &mut TasksBuilder) -> Self::Nodes {
        todo!();
    }

    /*
    fn new_source(nodes: &Self::Nodes) -> Self::Source {
        let mut vec = Vec::<T::Source>::new();

        for node in nodes {
            vec.push(T::new_source(node))
        }

        vec
    }
    */

    /*
    fn is_available(nodes: &Self::Nodes, data: &GraphData) -> bool {
        for node in nodes {
            if ! T::is_available(node, data) {
                return false;
            }
        }

        true
    }

    fn read(nodes: &Self::Nodes, data: &mut GraphData) -> Self {
        let mut vec = Vec::new();

        for node in nodes {
            vec.push(T::read(node, data));
        }

        vec
    }

    fn fill_input(
        source: &mut Self::Source, 
        nodes: &Self::Nodes, 
        data: &mut GraphData
    ) -> bool {
        for (source, node) in source.iter_mut().zip(nodes) {
            if ! T::fill_input(source, node, data) {
                return false
            }
        }

        true
    }

    fn write(nodes: &Self::Nodes, data: &mut GraphData, value: Self) -> bool {
        let mut value = value;
        for (value, node) in value.drain(..).zip(nodes) {
            T::write(node, data, value);
        }

        false
    }
    */
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

            /*
            fn new_source(nodes: &Self::Nodes) -> Self::Source {
                let ($($t),*) = nodes;

                ($(
                    $t::new_source($t)
                ),*)
            }
            */
        
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

            /*
            fn is_available(nodes: &Self::Nodes, data: &GraphData) -> bool {
                let ($($v),*) = nodes;

                $(
                    $t::is_available($v, data)
                )&&*
            }

            fn read(nodes: &Self::Nodes, data: &mut GraphData) -> Self {
                let ($($v),*) = nodes;

                ($(
                    $t::read($v, data)
                ),*)
            }

            fn fill_input(
                source: &mut Self::Source, 
                nodes: &Self::Nodes, 
                data: &mut GraphData
            ) -> bool {
                let ($($t),*) = nodes;
                let ($($v),*) = source;
        
                $(
                    if ! $t::fill_input($v, $t, data) {
                        return false
                    }
                )*

                true
            }
        
            fn write(nodes: &Self::Nodes, data: &mut GraphData, value: Self) -> bool {
                #[allow(non_snake_case)]
                let ($($t),*) = nodes;
                let ($($v),*) = value;

                $(
                    $t::write($t, data, $v);
                )*

                true
            }
            */


            fn fill_source(
                source: &mut Self::Source, 
            ) -> bool {
                let ($($v),*) = source;
        
                $(
                    if ! $t::fill_source($v) {
                        return false
                    }
                )*

                true
            }

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

            fn add_source(
                dst_id: NodeId,
                src_nodes: &Self::Nodes, 
                graph: &mut Graph,
                tasks: &mut TasksBuilder
            ) -> Self::Source {
                let ($($t),*) = src_nodes;

                ($(
                    $t::add_source(dst_id, $t, graph, tasks)
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
