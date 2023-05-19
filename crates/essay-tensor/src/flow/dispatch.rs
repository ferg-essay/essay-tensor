use super::{graph::NodeId, task::Tasks};

pub struct Dispatcher {
    active: Vec<NodeId>,
    request: Vec<SourceRequest>,
    wake: Vec<NodeId>,
    complete: Vec<NodeId>,
}

#[derive(Debug)]
pub struct SourceRequest(NodeId, usize, u64);

impl Dispatcher {
    pub fn new() -> Self {
        Self {
            active: Default::default(),
            request: Default::default(),
            wake: Default::default(),
            complete: Default::default(),
        }
    }

    pub fn spawn(&mut self, node: NodeId) {
        self.active.push(node);
        if self.active.len() > 1 {
            panic!("Unknown spawn");
        }
    }

    pub(crate) fn add_update(&mut self, id: NodeId) {
        self.wake.push(id);
    }

    pub fn request(&mut self, src_id: NodeId, src_index: usize, n_request: u64) {
        self.request.push(SourceRequest(src_id, src_index, n_request));
    }

    pub fn update(
        &mut self, 
        tasks: &mut Tasks, 
        // data: &mut GraphData, 
    ) -> bool {
        let mut is_update = false;

        let complete : Vec<NodeId> = self.complete.drain(..).collect();
        for id in complete {
            tasks.complete(id, self);
            is_update = true;
        }

        let wake : Vec<NodeId> = self.wake.drain(..).collect();
        for id in wake {
            tasks.update(id, self);
            is_update = true;
        }

        let requests : Vec<SourceRequest> = self.request.drain(..).collect();
        for request in requests {
            tasks.request(
                request.src_id(), 
                request.src_index(), 
                request.n_request(), 
                self
            );

            is_update = true;
        }

        while let Some(id) = self.active.pop() {
            is_update = true;

            tasks.execute(id, self);
        }

        is_update
    }

    pub fn complete(&mut self, node: NodeId) { // , _data: &mut GraphData) {
        self.complete.push(node);
    }
}


impl SourceRequest {
    pub(crate) fn src_id(&self) -> NodeId {
        self.0
    }

    pub(crate) fn src_index(&self) -> usize {
        self.1
    }

    pub(crate) fn n_request(&self) -> u64 {
        self.2
    }
}
