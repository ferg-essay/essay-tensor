use super::{graph::NodeId, task::Tasks};

pub struct Dispatcher {
    ready: Vec<NodeId>,
    wake: Vec<NodeId>,
}

impl Dispatcher {
    pub fn new() -> Self {
        Self {
            ready: Default::default(),
            wake: Default::default(),
        }
    }

    pub fn spawn(&mut self, node: NodeId) {
        self.ready.push(node);
    }

    pub fn dispatch(
        &mut self, 
        tasks: &mut Tasks, 
        // data: &mut GraphData
    ) -> bool {
        let mut is_active = false;

        while let Some(id) = self.ready.pop() {
            is_active = true;

            tasks.execute(id, self);

            /*
            let node = tasks.node_mut(id);
            
            node.execute(self).unwrap();
            */
        }

        is_active
    }

    pub fn wake(
        &mut self, 
        tasks: &mut Tasks, 
        // data: &mut GraphData, 
    ) {
        let wake : Vec<NodeId> = self.wake.drain(..).collect();

        for id in wake {
            tasks.update(id, self);
        }
    }

    pub fn complete(&mut self, node: NodeId) { // , _data: &mut GraphData) {
        self.wake.push(node);
    }
}
