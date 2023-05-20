use super::{task::{Tasks, NodeId}};

pub struct Dispatcher {
    commands: Vec<Wake>,
}

#[derive(Debug)]
pub struct SourceRequest(NodeId, usize, u64);

#[derive(Debug)]
enum Wake {
    RequestSource(NodeId, usize, u64),
    ReadySource(NodeId, usize, u64),
    Execute(NodeId),
    Complete(NodeId, bool),
}

impl Dispatcher {
    pub fn new() -> Self {
        Self {
            commands: Default::default(),
        }
    }

    pub fn request_source(&mut self, src_id: NodeId, sink_index: usize, n_request: u64) {
        self.commands.push(Wake::RequestSource(src_id, sink_index, n_request));
    }

    pub(crate) fn ready_source(&mut self, id: NodeId, source_index: usize, n_ready: u64) {
        self.commands.push(Wake::ReadySource(id, source_index, n_ready));
    }

    pub fn execute(&mut self, node: NodeId) {
        self.commands.push(Wake::Execute(node));
    }

    pub fn complete(&mut self, node: NodeId, is_done: bool) {
        self.commands.push(Wake::Complete(node, is_done));
    }

    pub fn apply(
        &mut self, 
        tasks: &mut Tasks, 
    ) -> bool {
        let mut is_update = false;

        while let Some(command) = self.commands.pop() {
            is_update = true;

            // println!("Command: {:?}", command);

            match command {
                Wake::RequestSource(src_id, sink_index, n_request) => {
                    tasks.request_source(src_id, sink_index, n_request, self);
                },
                Wake::ReadySource(dst_id, source_index, n_ready) => {
                    tasks.ready_source(dst_id, source_index, n_ready, self);
                },
                Wake::Execute(id) => {
                    tasks.execute(id, self);
                },
                Wake::Complete(id, is_done) => {
                    tasks.complete(id, is_done, self);
                },
            }
        }

        is_update
    }
}
