use std::{any::TypeId, mem::{self, ManuallyDrop}, ptr::NonNull, alloc::Layout};

use super::{
    task::{InputTask, self}, 
    flow::{TaskGraph}, dispatch::Dispatcher, graph::{TaskIdBare, TaskId, Graph}
};

pub enum Out<T> {
    None,
    Some(T),
    Pending,
}

pub trait FlowData {}

pub trait FlowIn<T> : Clone + 'static {
    type Nodes : Clone; //  : FlowNodes;
    type Source; // : FlowSource;

    fn add_arrows(
        id: TaskIdBare, 
        nodes_in: Self::Nodes, 
        arrows_in: &mut Vec<TaskIdBare>,
        graph: &mut Graph
    );

    fn node_ids(nodes: &Self::Nodes, ids: &mut Vec<TaskIdBare>);

    fn new_input(graph: &mut Graph, tasks: &mut TaskGraph) -> Self::Nodes;
    fn new_source(nodes: &Self::Nodes) -> Self::Source;

    fn wake(
        nodes: &Self::Nodes, 
        graph: &Graph,
        tasks: &mut TaskGraph,
        dispatcher: &mut Dispatcher,
        data: &mut GraphData,
    ) -> Out<()>;

    fn is_available(nodes: &Self::Nodes, data: &GraphData) -> bool;
    fn read(nodes: &Self::Nodes, data: &mut GraphData) -> T;
    fn fill_input(source: &mut Self::Source, nodes: &Self::Nodes, data: &mut GraphData) -> bool;

    fn write(nodes: &Self::Nodes, data: &mut GraphData, value: T) -> bool;
}

/*
pub trait FlowNodes : Clone + 'static {
    fn add_arrows(&self, node: TaskIdBare, graph: &mut TaskGraph);
}
*/
/*
pub trait FlowSource : 'static {
    type Item;
}
*/

pub struct GraphData {
    tasks: Vec<RawData>,
}

pub struct RawData {
    type_id: TypeId,
    layout: Layout,

    item: Option<NonNull<u8>>,
    n_arrows: usize,
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

impl RawData {
    pub fn new<T: Clone + 'static>(n_arrows: usize) -> Self {
        Self {
            type_id: TypeId::of::<T>(),
            layout: Layout::new::<T>(),
            n_arrows: n_arrows,
            item: None,
        }
    }

    #[inline]
    fn is_available(&self) -> bool {
        self.item.is_some()
    }

    fn read<T: Clone + 'static>(&mut self) -> T {
        if self.n_arrows > 1 {
            self.n_arrows -= 1;

            match self.item {
                Some(data) => unsafe {
                    data.cast::<T>().as_ref().clone()
                },
                None => panic!("expected value"),
            }
        } else if self.n_arrows == 1 {
            self.n_arrows -= 1;

            match self.item.take() {
                Some(data) => unsafe {
                    data.as_ptr().cast::<T>().read()
                },
                None => panic!("expected value"),
            }
        } else {
            panic!("unexpected number of arrows")
        }
    }

    fn write<T: 'static>(&mut self, data: T) -> bool {
        assert!(self.item.is_none());

        if self.n_arrows > 0 {
            self.item.replace(self.wrap(data));
        }

        false
    }

    fn wrap<T: 'static>(&self, item: T) -> NonNull::<u8> {
        assert!(self.type_id == TypeId::of::<T>());

        let layout = self.layout;

        unsafe {
            let data = std::alloc::alloc(layout);
            let data = NonNull::new(data).unwrap();

            let mut value = ManuallyDrop::new(item);
            let source = NonNull::from(&mut *value).cast();

            std::ptr::copy_nonoverlapping::<u8>(
                source.as_ptr(), 
                data.as_ptr(),
                mem::size_of::<T>(),
            );

            data
        }
    }
}

impl<T> Out<T> {
    #[inline]
    pub fn is_none(&self) -> bool {
        match self {
            Out::None => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_pending(&self) -> bool {
        match self {
            Out::Pending => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_some(&self) -> bool {
        match self {
            Out::Some(_) => true,
            _ => false,
        }
    }

    #[inline]
    pub fn take(&mut self) -> Self {
        match self {
            Out::None => Out::None,
            Out::Some(_) => mem::replace(self, Out::Pending),
            Out::Pending => Out::Pending,
        }
    }

    #[inline]
    pub fn replace(&mut self, value: Self) -> Self {
        mem::replace(self, value)
    }

    #[inline]
    pub fn unwrap(&mut self) -> T {
        if let Out::Some(_) = self {
            let v = mem::replace(self, Out::Pending);
            if let Out::Some(v) = v {
                return v
            }
        }

        panic!("Unwrap with invalid value")
    }
}

impl<T> Default for Out<T> {
    fn default() -> Self {
        Out::None
    }
}

impl FlowIn<()> for () {
    type Nodes = TaskId<()>;
    type Source = ();

    fn add_arrows(
        id: TaskIdBare,
        node_in: Self::Nodes, 
        arrows_in: &mut Vec<TaskIdBare>, 
        graph: &mut Graph
    ) {
        arrows_in.push(node_in.id());

        graph.add_arrow_out(node_in.id(), id);
    }

    fn node_ids(
        nodes: &Self::Nodes,
        ids: &mut Vec<TaskIdBare>, 
    ) {
        ids.push(nodes.id());
    }

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

    fn new_input(graph: &mut Graph, tasks: &mut TaskGraph) -> Self::Nodes {
        let id = graph.push_input::<()>();

        let node = InputTask::<()>::new(id);

        tasks.push_node(Box::new(node));

        id
    }

    fn new_source(_nodes: &Self::Nodes) -> Self::Source {
        ()
    }

    fn wake(
        _nodes: &Self::Nodes, 
        _graph: &Graph,
        _tasks: &mut TaskGraph,
        _dispatcher: &mut Dispatcher,
        _data: &mut GraphData,
    ) -> Out<()> {
        Out::Some(())
    }
}

impl<T:FlowData + Clone + 'static> FlowIn<T> for T {
    type Nodes = TaskId<T>;
    type Source = task::Source<T>;

    fn node_ids(
        nodes: &Self::Nodes,
        ids: &mut Vec<TaskIdBare>, 
    ) {
        ids.push(nodes.id());
    }

    fn add_arrows(
        id: TaskIdBare,
        node_in: Self::Nodes, 
        arrows_in: &mut Vec<TaskIdBare>, 
        graph: &mut Graph
    ) {
        arrows_in.push(node_in.id());

        graph.add_arrow_out(node_in.id(), id);
    }

    fn new_input(graph: &mut Graph, task_graph: &mut TaskGraph) -> Self::Nodes {
        let id = graph.push_input::<T>();

        let node = InputTask::<T>::new(id.clone());

        task_graph.push_node(Box::new(node));

        id
    }

    fn new_source(_nodes: &Self::Nodes) -> Self::Source {
        task::Source::new()
    }

    fn wake(
        nodes: &Self::Nodes, 
        graph: &Graph,
        tasks: &mut TaskGraph,
        dispatcher: &mut Dispatcher,
        data: &mut GraphData,
    ) -> Out<()> {
        graph.wake(&nodes.id(), tasks, dispatcher, data)
    }

    fn is_available(nodes: &Self::Nodes, data: &GraphData) -> bool {
        data.is_available(nodes)
    }

    fn read(nodes: &Self::Nodes, data: &mut GraphData) -> Self {
        data.read(nodes)
    }

    fn fill_input(
        source: &mut Self::Source, 
        nodes: &Self::Nodes, 
        data: &mut GraphData
    ) -> bool {
        if source.is_some() {
            true
        } else if source.is_none() {
            false
        } else if data.is_available(nodes) {
            source.push(data.read(nodes));

            true
        } else {
            false
        }
    }

    fn write(nodes: &Self::Nodes, data: &mut GraphData, value: Self) -> bool {
        data.write(nodes, value)
    }
}

impl<T: FlowIn<T>> FlowIn<Vec<T>> for Vec<T> {
    type Nodes = Vec<T::Nodes>;
    type Source = Vec<T::Source>;

    fn node_ids(
        nodes: &Self::Nodes,
        ids: &mut Vec<TaskIdBare>, 
    ) {
        for node_in in nodes {
            T::node_ids(node_in, ids);
        }
    }

    fn add_arrows(
        id: TaskIdBare,
        nodes_in: Self::Nodes, 
        arrows_in: &mut Vec<TaskIdBare>, 
        graph: &mut Graph
    ) {
        for node_in in nodes_in {
            T::add_arrows(id, node_in, arrows_in, graph)
        }
    }

    fn new_input(_graph: &mut Graph, _tasks: &mut TaskGraph) -> Self::Nodes {
        todo!();
        /*
        let id = graph.alloc_id::<Vec<T>>();
        let node = InputNode::<Vec<T>>::new(id.clone());

        graph.push_node(Box::new(node));

        id
        */
    }

    fn new_source(nodes: &Self::Nodes) -> Self::Source {
        let mut vec = Vec::<T::Source>::new();

        for node in nodes {
            vec.push(T::new_source(node))
        }

        vec
    }

    fn wake(
        nodes: &Self::Nodes, 
        graph: &Graph,
        tasks: &mut TaskGraph,
        dispatcher: &mut Dispatcher,
        data: &mut GraphData,
    ) -> Out<()> {
        let mut out = Out::Some(());

        for node in nodes {
            match T::wake(node, graph, tasks, dispatcher, data) {
                Out::None => return Out::None,
                Out::Pending => out = Out::Pending,
                Out::Some(_) => {},
            }
        }

        out
    }

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

            fn node_ids(
                nodes: &Self::Nodes,
                ids: &mut Vec<TaskIdBare>, 
            ) {
                let ($($t),*) = nodes;

                $(
                    $t::node_ids($t, ids);
                )*
            }
        
            fn add_arrows(
                id: TaskIdBare,
                nodes_in: Self::Nodes, 
                arrows_in: &mut Vec<TaskIdBare>, 
                graph: &mut Graph
            ) {
                let ($($t),*) = nodes_in;

                $(
                    $t::add_arrows(id, $t, arrows_in, graph);
                )*
            }
        
            fn new_input(graph: &mut Graph, tasks: &mut TaskGraph) -> Self::Nodes {
                let key = ($(
                    $t::new_input(graph, tasks)
                ),*);

                let task = InputTask::<($($t),*)>::new(key.clone());

                tasks.push_node(Box::new(task));

                key
            }

            fn new_source(nodes: &Self::Nodes) -> Self::Source {
                let ($($t),*) = nodes;

                ($(
                    $t::new_source($t)
                ),*)
            }

            fn wake(
                nodes: &Self::Nodes, 
                graph: &Graph,
                tasks: &mut TaskGraph,
                dispatcher: &mut Dispatcher,
                data: &mut GraphData,
            ) -> Out<()> {
                let ($($t),*) = nodes;

                let mut out = Out::Some(());
        
                $(
                    match $t::wake($t, graph, tasks, dispatcher, data) {
                        Out::None => return Out::None,
                        Out::Pending => out = Out::Pending,
                        Out::Some(_) => {},
                    };
                )*
        
                out
            }
        
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
/*
impl FlowNodes for () {
    fn add_arrows(&self, _node: TaskIdBare, _graph: &mut TaskGraph) {
    }
}

impl<T: 'static> FlowNodes for TaskId<T> {
    fn add_arrows(&self, node: TaskIdBare, graph: &mut TaskGraph) {
        graph.add_arrow(self.id(), node);
    }
}

impl<T: FlowNodes> FlowNodes for Vec<T> {
    fn add_arrows(&self, node: TaskIdBare, graph: &mut TaskGraph) {
        for id in self {
            id.add_arrows(node, graph);
        }
    }
}

macro_rules! tuple_nodes {
    ($($id:ident),*) => {
        #[allow(non_snake_case)]
        impl<$($id),*> FlowNodes for ($($id),*)
        where
            $($id: FlowNodes),*
        {
            fn add_arrows(&self, node: TaskIdBare, graph: &mut Graph) {
                let ($($id),*) = &self;

                $($id.add_arrows(node, graph));*
            }
        }
    }
}

tuple_nodes!(T1, T2);
tuple_nodes!(T1, T2, T3);
tuple_nodes!(T1, T2, T3, T4);
tuple_nodes!(T1, T2, T3, T4, T5);
tuple_nodes!(T1, T2, T3, T4, T5, T6);
tuple_nodes!(T1, T2, T3, T4, T5, T6, T7);
tuple_nodes!(T1, T2, T3, T4, T5, T6, T7, T8);
*/
/*
impl FlowSource for () {
    type Item = ();
}

impl<T> FlowSource for Source<T>
where
    T: FlowData + Clone + 'static,
{
    type Item = T;
}

impl<T> FlowSource for Vec<T>
where
    T: FlowSource,
{
    type Item = Vec<T::Item>;
}

macro_rules! tuple_source {
    ($($t:ident),*) => {
        #[allow(non_snake_case)]
        impl<$($t),*> FlowSource for ($($t),*)
        where
            $($t: FlowSource),*,
        {
            type Item = ($($t::Item),*);
        }
    }
}

tuple_source!(T1, T2);
tuple_source!(T1, T2, T3);
tuple_source!(T1, T2, T3, T4);
tuple_source!(T1, T2, T3, T4, T5);
tuple_source!(T1, T2, T3, T4, T5, T6);
tuple_source!(T1, T2, T3, T4, T5, T6, T7);
tuple_source!(T1, T2, T3, T4, T5, T6, T7, T8);
*/

// impl FlowData for () {}
impl FlowData for String {}
impl FlowData for usize {}
impl FlowData for i32 {}
impl FlowData for f32 {}
