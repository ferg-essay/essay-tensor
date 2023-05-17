use std::marker::PhantomData;

use super::{data::{FlowIn, FlowData}, flow::TaskGraph};

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct TaskIdBare(usize);

#[derive(Copy, Debug, PartialEq)]
pub struct TaskId<T> {
    index: usize,
    marker: PhantomData<T>,
}


pub struct Graph {
    nodes: Vec<Node>,
}

struct Node {
    id: TaskIdBare,

    arrows_in: Vec<TaskIdBare>,
    arrows_out: Vec<TaskIdBare>,
}

impl Node {
    fn new(id: TaskIdBare, in_arrows: Vec<TaskIdBare>) -> Self {
        Self {
            id,
            arrows_in: in_arrows,
            arrows_out: Default::default(),
        }
    }

    fn add_output_arrow(&mut self, id: TaskIdBare) {
        self.arrows_out.push(id);
    }
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_input<I>(&mut self) -> TaskId<I> 
    where
        I: 'static
    {
        let id = TaskId::<I>::new(self.nodes.len());

        let mut arrows_in = Vec::<TaskIdBare>::new();

        let node = Node::new(id.id(), arrows_in);

        self.nodes.push(node);

        id
    }

    pub fn push<I, O>(&mut self, nodes_in: I::Nodes) -> TaskId<O> 
    where
        I: FlowIn<I>,
        O: 'static
    {
        let id = TaskId::<O>::new(self.nodes.len());

        let mut arrows_in = Vec::<TaskIdBare>::new();

        I::add_arrows(id.id(), nodes_in, &mut arrows_in, self);

        let node = Node::new(id.id(), arrows_in);

        self.nodes.push(node);

        // in_arrows.add_arrows(id, &mut self);

        id
    }

    pub fn add_arrow_out(&mut self, src: TaskIdBare, dst: TaskIdBare)
    {
        self.nodes[src.index()].arrows_out.push(dst);
    }

    pub(crate) fn wake<T: FlowIn<T>>(
        &self, 
        nodes: T::Nodes, 
        tasks: &mut TaskGraph, 
        dispatcher: &mut super::dispatch::Dispatcher, 
        data: &mut super::data::GraphData
    ) -> bool {
        todo!()
    }

}

impl Default for Graph {
    fn default() -> Self {
        Self { nodes: Default::default() }
    }
}

impl<T> Clone for TaskId<T> {
    fn clone(&self) -> Self {
        Self { 
            index: self.index,
            marker: self.marker.clone() 
        }
    }
}

impl TaskIdBare {
    pub fn index(&self) -> usize {
        self.0
    }
}

impl<T: 'static> TaskId<T> {
    fn new(index: usize) -> Self {
        Self {
            index,
            marker: PhantomData,
        }
    }

    #[inline]
    pub fn id(&self) -> TaskIdBare {
        TaskIdBare(self.index)
    }

    #[inline]
    pub fn index(&self) -> usize {
        self.index
    }
}
