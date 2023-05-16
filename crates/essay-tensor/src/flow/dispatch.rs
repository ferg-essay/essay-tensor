use super::{flow::{Graph, TaskId}, data::GraphData};

pub struct BasicDispatcher {
    ready: Vec<TaskId>,
}

pub trait Dispatcher {
    fn spawn(&mut self, node: TaskId);
}

pub struct Waker {
    nodes: Vec<TaskId>,
}

impl BasicDispatcher {
    pub fn new() -> Self {
        Self {
            ready: Default::default(),
        }
    }

    pub fn dispatch(
        &mut self, 
        graph: &mut Graph, 
        waker: &mut Waker,
        data: &mut GraphData
    ) -> bool {
        let mut is_active = false;

        while let Some(id) = self.ready.pop() {
            is_active = true;

            let node = graph.node_mut(id);
            
            node.execute(data, waker);
        }

        is_active
    }
}

impl Dispatcher for BasicDispatcher {
    fn spawn(&mut self, node: TaskId) {
        self.ready.push(node);
        println!("Spawn: {:?}", node);
    }
}

impl Waker {
    pub fn new() -> Self {
        Self {
            nodes: Default::default(),
        }
    }

    pub fn wake(
        &mut self, 
        graph: &mut Graph, 
        data: &mut GraphData, 
        dispatcher: &mut dyn Dispatcher
    ) {
        for id in self.nodes.drain(..) {
            graph.node_mut(id).update(data, dispatcher);
        }
    }

    pub fn complete(&mut self, node: TaskId, _data: &mut GraphData) {
        self.nodes.push(node);
        println!("Complete");
    }
}
