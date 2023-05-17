use core::fmt;
use std::marker::PhantomData;

use super::{data::{FlowIn, Out}, flow::TaskGraph};

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct TaskIdBare(usize);

#[derive(Copy, PartialEq)]
pub struct TaskId<T> {
    index: usize,
    marker: PhantomData<T>,
}

pub struct Graph {
    nodes: Vec<Node>,
}

#[derive(Debug)]
struct Node {
    _id: TaskIdBare,

    arrows_in: Vec<TaskIdBare>,
    arrows_out: Vec<TaskIdBare>,
}

impl Node {
    fn new(id: TaskIdBare, in_arrows: Vec<TaskIdBare>) -> Self {
        Self {
            _id: id,
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

        let arrows_in = Vec::<TaskIdBare>::new();

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
        self.nodes[src.index()].add_output_arrow(dst);
    }

    pub(crate) fn wake(
        &self, 
        id: &TaskIdBare,
        tasks: &mut TaskGraph, 
        dispatcher: &mut super::dispatch::Dispatcher, 
        data: &mut super::data::GraphData
    ) -> Out<()> {
        let task = tasks.node_mut(*id);

        match task.wake(dispatcher, data) {
            Out::None => Out::None,
            Out::Some(_) => Out::Some(()),
            Out::Pending => {
                for node_id in &self.nodes[id.index()].arrows_in {
                    if self.wake(node_id, tasks, dispatcher, data).is_none() {
                        return Out::None;
                    }
                }

                Out::Some(())
            }
        }
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self { nodes: Default::default() }
    }
}

impl<T> fmt::Debug for TaskId<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TaskId[{}]", self.index)
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
