use super::{
    source::{NodeId, SourcesOuter, SourcesInner}, 
    thread_pool::{Child, Result, Sender, Msg, Parent, MainId}};

pub trait InnerWaker {
    fn post_execute(&mut self, task: NodeId, id_done: bool);

    fn data_request(&mut self, dst_id: NodeId, out_index: usize, n_requested: u64);

    fn data_ready(&mut self, dst_id: NodeId, input_index: usize, n_sent: u64);
}

pub trait OuterWaker {
    fn execute(&mut self, task: NodeId);

    fn data_request(&mut self, dst_id: NodeId, out_index: usize, n_requested: u64);
}

//
// Parent (dispatcher) task manager
//

#[derive(Debug)]
enum MainRequest {
    Next(NodeId),
}

impl Msg for MainRequest {}

#[derive(Debug)]
enum MainReply {
    Complete,
}

impl Msg for MainReply {}

pub(crate) struct FlowParent {
    tasks: SourcesOuter,
    // waker: DispatcherOuter,
}

impl Parent<MainRequest, MainReply, ChildRequest, ChildReply> for FlowParent {
    fn on_start(&mut self) {
    }

    fn on_main_start(&mut self, id: super::thread_pool::MainId) {
    }

    fn on_main(
        &mut self, 
        _main_id: MainId, 
        msg: MainRequest, 
        to_main: &mut dyn Sender<MainReply>, 
        to_child: &mut dyn Sender<ChildRequest>
    ) -> Result<()> {
        match msg {
            MainRequest::Next(output_id) => todo!(),
        }
    }

    fn on_main_end(&mut self, id: super::thread_pool::MainId) {
    }

    fn on_child(
        &mut self, 
        msg: ChildReply, 
        to_main: &mut dyn Sender<MainReply>, 
        to_child: &mut dyn Sender<ChildRequest>
    ) -> Result<()> {
        let mut waker = ParentWaker::new(to_child);

        match msg {
            ChildReply::PostExecute(node_id, is_done) => {
                self.tasks.post_execute(node_id, is_done, &mut waker);
            }
            ChildReply::DataRequest(src_id, sink_index, n_request) => {
                self.tasks.data_request(src_id, sink_index, n_request, &mut waker);
            }
            ChildReply::DataReady(dst_id, source_index, n_ready) => {
                self.tasks.data_ready(dst_id, source_index, n_ready, &mut waker);
            }
        }

        // waker.apply(&mut self.tasks);
        Ok(())
    }
}

pub struct ParentWaker<'a> {
    to_child: &'a mut dyn Sender<ChildRequest>,
}

impl<'a> ParentWaker<'a> {
    fn new(
        to_child: &'a mut dyn Sender<ChildRequest>,
    ) -> Self {
        Self {
            to_child
        }
    }
}

impl OuterWaker for ParentWaker<'_> {
    fn data_request(&mut self, dst_id: NodeId, out_index: usize, n_requested: u64) {
        todo!()
    }

    fn execute(&mut self, task: NodeId) {
        self.to_child.send(ChildRequest::Execute(task)).unwrap();
    }
}

//
// Child task manager
//

#[derive(Debug)]
enum ChildRequest {
    Execute(NodeId),
}

impl Msg for ChildRequest {}

#[derive(Debug)]
enum ChildReply {
    PostExecute(NodeId, bool),
    DataRequest(NodeId, usize, u64),
    DataReady(NodeId, usize, u64),
}

impl Msg for ChildReply {}

pub(crate) struct FlowChild {
    tasks: SourcesInner,
}

impl Child<ChildRequest, ChildReply> for FlowChild {
    fn on_start(&mut self) {
    }

    fn on_parent(
        &mut self, 
        msg: ChildRequest, 
        to_parent: &mut dyn Sender<ChildReply>
    ) -> Result<()> {
        match msg {
            ChildRequest::Execute(node_id) => {
                let mut waker = ChildWaker::new(to_parent);
                self.tasks.execute(node_id, &mut waker);
                Ok(())
            }
        }
    }
}

pub struct ChildWaker<'a> {
    to_parent: &'a mut dyn Sender<ChildReply>,
}

impl<'a> ChildWaker<'a> {
    fn new(to_parent: &'a mut dyn Sender<ChildReply>) -> Self {
        Self {
            to_parent
        }
    }
}

impl InnerWaker for ChildWaker<'_> {
    fn post_execute(&mut self, task: NodeId, is_done: bool) {
        self.to_parent.send(ChildReply::PostExecute(task, is_done)).unwrap();
    }

    fn data_ready(&mut self, dst_id: NodeId, input_index: usize, n_sent: u64) {
        self.to_parent.send(ChildReply::DataReady(dst_id, input_index, n_sent)).unwrap();
    }

    fn data_request(&mut self, dst_id: NodeId, out_index: usize, n_requested: u64) {
        todo!()
    }
}

//
// SingleDispatcher
//

#[derive(Debug)]
pub struct SourceRequest(NodeId, usize, u64);

#[derive(Debug)]
enum Wake {
    RequestSource(NodeId, usize, u64),
    ReadySource(NodeId, usize, u64),
    Execute(NodeId),
    Complete(NodeId, bool),
}

pub struct Dispatcher {
    outer_commands: Vec<Wake>,
    inner_commands: Vec<Wake>,
}

impl Dispatcher {
    pub fn new<'a>() -> Self {
        Self {
            outer_commands: Vec::new(),
            inner_commands: Vec::new(),
        }
    }

    pub fn outer(&mut self) -> DispatcherOuter {
        DispatcherOuter::new(self)
    }

    pub fn inner(&mut self) -> DispatcherInner {
        DispatcherInner::new(self)
    }
}

pub struct DispatcherOuter<'a> {
    outer_commands: &'a mut Vec<Wake>,
    inner_commands: &'a mut Vec<Wake>,
}

impl<'a> DispatcherOuter<'a> {
    pub fn new(dispatcher: &'a mut Dispatcher) -> Self {
        Self {
            outer_commands: &mut dispatcher.outer_commands,
            inner_commands: &mut dispatcher.inner_commands,
        }
    }

    pub fn apply(
        &mut self, 
        tasks: &mut SourcesOuter, 
    ) -> bool {
        let mut is_update = false;

        while let Some(command) = self.outer_commands.pop() {
            is_update = true;

            // println!("Command: {:?}", command);

            match command {
                Wake::RequestSource(src_id, sink_index, n_request) => {
                    tasks.data_request(src_id, sink_index, n_request, self);
                },
                Wake::ReadySource(dst_id, source_index, n_ready) => {
                    tasks.data_ready(dst_id, source_index, n_ready, self);
                },
                Wake::Complete(id, is_done) => {
                    tasks.post_execute(id, is_done, self);
                },
                _ => { 
                    todo!();
                }
            }
        }

        is_update
    }
}

impl OuterWaker for DispatcherOuter<'_> {
    fn execute(&mut self, task: NodeId) {
        self.inner_commands.push(Wake::Execute(task));
        todo!()
    }

    fn data_request(&mut self, src_id: NodeId, out_index: usize, n_request: u64) {
        self.outer_commands.push(Wake::RequestSource(src_id, out_index, n_request));
    }
}

pub struct DispatcherInner<'a> {
    outer_commands: &'a mut Vec<Wake>,
    inner_commands: &'a mut Vec<Wake>,
}

impl<'a> DispatcherInner<'a> {
    pub fn new(dispatcher: &'a mut Dispatcher) -> Self {
        Self {
            outer_commands: &mut dispatcher.outer_commands,
            inner_commands: &mut dispatcher.inner_commands,
        }
    }

    pub fn execute(&mut self, node: NodeId) {
        self.inner_commands.push(Wake::Execute(node));
    }

    pub fn apply(
        &mut self, 
        tasks: &mut SourcesInner, 
    ) -> bool {
        let mut is_update = false;

        while let Some(command) = self.inner_commands.pop() {
            is_update = true;

            // println!("Command: {:?}", command);

            match command {
                Wake::Execute(id) => {
                    tasks.execute(id, self);
                },
                _ => { todo!() }
            }
        }

        is_update
    }
}

impl InnerWaker for DispatcherInner<'_> {
    fn data_request(&mut self, src_id: NodeId, sink_index: usize, n_request: u64) {
        self.outer_commands.push(Wake::RequestSource(src_id, sink_index, n_request));
    }

    fn data_ready(&mut self, id: NodeId, source_index: usize, n_ready: u64) {
        self.outer_commands.push(Wake::ReadySource(id, source_index, n_ready));
    }

    fn post_execute(&mut self, node: NodeId, is_done: bool) {
        self.outer_commands.push(Wake::Complete(node, is_done));
    }
}
