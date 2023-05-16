use super::{flow::{Graph, TaskIdBare}, data::GraphData};

pub struct Dispatcher {
    ready: Vec<TaskIdBare>,
    wake: Vec<TaskIdBare>,
}

impl Dispatcher {
    pub fn new() -> Self {
        Self {
            ready: Default::default(),
            wake: Default::default(),
        }
    }

    pub fn spawn(&mut self, node: TaskIdBare) {
        self.ready.push(node);
    }

    pub fn dispatch(
        &mut self, 
        graph: &mut Graph, 
        data: &mut GraphData
    ) -> bool {
        let mut is_active = false;

        while let Some(id) = self.ready.pop() {
            is_active = true;

            let node = graph.node_mut(id);
            
            node.execute(data, self).unwrap();
        }

        is_active
    }

    pub fn wake(
        &mut self, 
        graph: &mut Graph, 
        data: &mut GraphData, 
    ) {
        let wake : Vec<TaskIdBare> = self.wake.drain(..).collect();

        for id in wake {
            graph.node_mut(id).update(data, self);
        }
    }

    pub fn complete(&mut self, node: TaskIdBare, _data: &mut GraphData) {
        self.wake.push(node);
    }
}
