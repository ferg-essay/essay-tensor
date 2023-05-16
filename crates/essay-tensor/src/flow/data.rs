use std::{any::TypeId, mem::{self, ManuallyDrop}, ptr::NonNull, alloc::Layout};

use super::{task::{InputNode}, flow::{TypedTaskId, FlowNodes, Graph}};

pub trait FlowData {}

pub trait FlowIn<T> : Clone + 'static {
    // type Item;
    type Nodes : FlowNodes;

    fn new_input(graph: &mut Graph) -> Self::Nodes;

    fn is_available(nodes: &Self::Nodes, data: &GraphData) -> bool;
    fn read(nodes: &Self::Nodes, data: &mut GraphData) -> T;
    fn write(nodes: &Self::Nodes, data: &mut GraphData, value: T) -> bool;
}

pub struct GraphData {
    nodes: Vec<RawData>,
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
            nodes: Vec::new(),
        }
    }

    pub fn push<T: Clone + 'static>(&mut self, n_arrows: usize) {
        self.nodes.push(RawData::new::<T>(n_arrows));
    }

    pub fn is_available<T: 'static>(&self, node: &TypedTaskId<T>) -> bool {
        self.nodes[node.index()].is_available()
    }

    pub fn read<T: Clone + 'static>(&mut self, node: &TypedTaskId<T>) -> T {
        self.nodes[node.index()].read()
    }
    
    pub fn write<T: 'static>(&mut self, node: &TypedTaskId<T>, data: T) -> bool {
        self.nodes[node.index()].write(data)
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

impl FlowIn<()> for () {
    type Nodes = ();

    fn is_available(_nodes: &Self::Nodes, _data: &GraphData) -> bool {
        true
    }

    fn read(_nodes: &Self::Nodes, _data: &mut GraphData) -> () {
        ()
    }

    fn write(_nodes: &Self::Nodes, _data: &mut GraphData, _value: ()) -> bool {
        false
    }

    fn new_input(_graph: &mut Graph) -> Self::Nodes {
        ()
    }
}

impl<T:FlowData + Clone + 'static> FlowIn<T> for T {
    type Nodes = TypedTaskId<T>;

    fn new_input(graph: &mut Graph) -> Self::Nodes {
        let id = graph.alloc_id::<T>();

        let node = InputNode::<T>::new(id.clone());

        graph.push_node(Box::new(node));

        id
    }

    fn is_available(nodes: &Self::Nodes, data: &GraphData) -> bool {
        data.is_available(nodes)
    }

    fn read(nodes: &Self::Nodes, data: &mut GraphData) -> Self {
        data.read(nodes)
    }

    fn write(nodes: &Self::Nodes, data: &mut GraphData, value: Self) -> bool {
        data.write(nodes, value)
    }
}

macro_rules! flow_tuple {
    ($(($t:ident, $v:ident)),*) => {

        impl<$($t),*> FlowIn<($($t),*)> for ($($t),*)
        where $(
            $t: FlowIn<$t>,
        )*
        {
            type Nodes = ($($t::Nodes),*);

            fn new_input(graph: &mut Graph) -> Self::Nodes {
                let key = ($(
                    $t::new_input(graph)
                ),*);

                let node = InputNode::<($($t),*)>::new(key.clone());

                graph.push_node(Box::new(node));

                key
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

flow_tuple!((T1, v1), (T2, v2));
flow_tuple!((T1, v1), (T2, v2), (T3, v3));
flow_tuple!((T1, v1), (T2, v2), (T3, v3), (T4, v4));
flow_tuple!((T1, v1), (T2, v2), (T3, v3), (T4, v4), (T5, v5));
flow_tuple!((T1, v1), (T2, v2), (T3, v3), (T4, v4), (T5, v5), (T6, v6));
flow_tuple!((T1, v1), (T2, v2), (T3, v3), (T4, v4), (T5, v5), (T6, v6), (T7, v7));
flow_tuple!((T1, v1), (T2, v2), (T3, v3), (T4, v4), (T5, v5), (T6, v6), (T7, v7), (T8, v8));

impl<T: FlowIn<T>> FlowIn<Vec<T>> for Vec<T> {
    type Nodes = Vec<T::Nodes>;

    fn new_input(_graph: &mut Graph) -> Self::Nodes {
        todo!();
        /*
        let id = graph.alloc_id::<Vec<T>>();
        let node = InputNode::<Vec<T>>::new(id.clone());

        graph.push_node(Box::new(node));

        id
        */
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

    fn write(nodes: &Self::Nodes, data: &mut GraphData, value: Self) -> bool {
        let mut value = value;
        for (value, node) in value.drain(..).zip(nodes) {
            T::write(node, data, value);
        }

        false
    }
}

//impl Scalar for () {}
impl FlowData for String {}
impl FlowData for usize {}
impl FlowData for i32 {}
impl FlowData for f32 {}
