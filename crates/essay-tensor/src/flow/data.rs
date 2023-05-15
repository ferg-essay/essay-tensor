use std::{any::TypeId, mem::{self, ManuallyDrop}, ptr::NonNull, alloc::Layout};

use super::{node::{Node}, flow::TypedTaskId};

pub trait Scalar {}

pub trait FlowData<T> : Clone + 'static {
    // type Item;
    type Nodes : Clone;

    fn read(nodes: &Self::Nodes, data: &mut GraphData) -> Option<T>;
    fn write(nodes: &Self::Nodes, data: &mut GraphData, value: T) -> bool;
}

pub struct GraphData {
    nodes: Vec<RawData>,
}

pub struct RawData {
    type_id: TypeId,
    layout: Layout,

    item: Option<NonNull<u8>>,
}

impl GraphData {
    pub fn new(nodes: &[Box<dyn Node>]) -> Self {
        let mut vec = Vec::new();

        for node in nodes {
            vec.push(node.new_data());
        }

        Self {
            nodes: vec,
        }
    }

    pub fn read<T: 'static>(&mut self, node: &TypedTaskId<T>) -> Option<T> {
        println!("Read {:?}", node.id());
        self.nodes[node.index()].read()
    }
    
    pub fn write<T: 'static>(&mut self, node: &TypedTaskId<T>, data: T) -> bool {
        println!("Write {:?}", node.id());
        self.nodes[node.index()].write(data)
    }
}

impl RawData {
    pub fn new<T: Clone + 'static>() -> Self {
        Self {
            type_id: TypeId::of::<T>(),
            layout: Layout::new::<T>(),
            item: None,
        }
    }

    fn read<T: 'static>(&mut self) -> Option<T> {
        match self.item.take() {
            Some(data) => Some(self.unwrap(data)),
            None => None,
        }
    }

    fn write<T: 'static>(&mut self, data: T) -> bool {
        assert!(self.item.is_none());

        self.item.replace(self.wrap(data));

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

    fn unwrap<T: 'static>(&self, data: NonNull<u8>) -> T {
        assert!(self.type_id == TypeId::of::<T>());

        unsafe { data.as_ptr().cast::<T>().read() }
    }
}

impl FlowData<()> for () {
    // type Item = ();
    type Nodes = ();

    fn read(_nodes: &Self::Nodes, _data: &mut GraphData) -> Option<()> {
        Some(())
    }

    fn write(_nodes: &Self::Nodes, _data: &mut GraphData, _value: ()) -> bool {
        false
    }
}

impl<T:Scalar + Clone + 'static> FlowData<T> for T {
    // type Item = T;
    type Nodes = TypedTaskId<T>;

    fn read(nodes: &Self::Nodes, data: &mut GraphData) -> Option<T> {
        data.read(nodes)
    }

    fn write(nodes: &Self::Nodes, data: &mut GraphData, value: T) -> bool {
        data.write(nodes, value)
    }
}

// pub trait FlowData : Send + 'static {}

//impl Scalar for () {}
impl Scalar for String {}
impl Scalar for usize {}
