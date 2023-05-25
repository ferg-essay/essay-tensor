use core::fmt;
use std::{mem, sync::{mpsc::{self}, Mutex, Arc}};

use super::{data::FlowData, dispatch::{InnerWaker}, source::{NodeId, SourceId, Out}};

pub trait PipeIn<T> : Send {
    fn out_index(&self) -> usize;

    fn init(&mut self);

    fn fill(&mut self, waker: &mut dyn InnerWaker) -> bool;
    fn n_read(&self) -> u64;

    fn next(&mut self) -> Option<T>;
}

pub trait PipeOut<T> : Send {
    fn dst_id(&self) -> NodeId;

    fn send(&mut self, value: Option<T>);
}

pub struct ChannelIn<T> {
    src_id: NodeId,
    src_index: usize,

    _dst: NodeId,
    _dst_index: usize,

    receiver: mpsc::Receiver<Option<T>>,
    n_receive: u64,

    value: Out<T>,
}

pub struct In<T>(Box<dyn PipeIn<T>>);

pub(crate) fn pipe<T: Send + 'static>(
    src_id: SourceId<T>, 
    src_index: usize,

    dst_id: NodeId,
    dst_index: usize,
) -> (Box<dyn PipeIn<T>>, Box<dyn PipeOut<T>>) {
    let (sender, receiver) = mpsc::channel::<Option<T>>();

    (
        Box::new(ChannelIn::new(src_id.id(), src_index, dst_id, dst_index, receiver)),
        Box::new(ChannelOut::new(src_id.id(), dst_id, sender))
    )
}

impl<T> In<T> {
    pub fn next(&mut self) -> Option<T> {
        self.0.next()
    }

    pub(crate) fn init(&mut self) {
        self.0.init();
    }

    pub(crate) fn n_read(&self) -> u64 {
        self.0.n_read()
    }
}

impl<T> In<T>
where
    T: FlowData + Clone + 'static
{
    pub(crate) fn new(input: Box<dyn PipeIn<T>>) -> Self {
        Self(input)
    }

    pub(crate) fn fill(&mut self, waker: &mut dyn InnerWaker) -> bool {
        self.0.fill(waker)
    }
}

//
// ChannelIn
//

impl<T> ChannelIn<T> {
    fn new(
        src_id: NodeId, 
        src_index: usize, 

        dst_id: NodeId, 
        dst_index: usize,

        receiver: mpsc::Receiver<Option<T>>
    ) -> Self {
        Self {
            src_id,
            src_index,

            _dst: dst_id,
            _dst_index: dst_index,

            receiver,
            value: Out::Pending,
            n_receive: 0,
        }
    }
}

impl<T: Send + 'static> PipeIn<T> for ChannelIn<T> {
    fn out_index(&self) -> usize {
        self.src_index
    }

    fn init(&mut self) {
        self.value.replace(Out::Pending);

        self.n_receive = 0;

        while let Ok(_) = self.receiver.try_recv() {
        }
    }

    fn next(&mut self) -> Option<T> {
        match self.value.take() {
            Out::None => None,
            Out::Some(value) => Some(value),
            Out::Pending => {
                panic!("{:?}.next() can't be called twice in a task", self);
            }
        }
    }

    fn fill(&mut self, waker: &mut dyn InnerWaker) -> bool {
        if ! self.value.is_pending() {
            true
        } else {
            match self.receiver.try_recv() {
                Ok(Some(value)) => { 
                    self.value.replace(Out::Some(value)); 
                    self.n_receive += 1;
                    true 
                }

                Ok(None) => { 
                    self.value.replace(Out::None);
                    self.n_receive += 1;
                    true 
                }

                Err(err) => {
                    match err {
                        mpsc::TryRecvError::Empty => {
                            waker.data_request(self.src_id, self.src_index, self.n_receive + 1);
                            false
                        },
                        mpsc::TryRecvError::Disconnected => {
                            panic!("Source channel unexpected disconnect");
                        }
                    }
                }
            }
        }
    }

    fn n_read(&self) -> u64 {
        self.n_receive
    }
} 

//
// ChannelOut
//

pub struct ChannelOut<T> {
    _src: NodeId,
    _dst: NodeId,

    sender: mpsc::Sender<Option<T>>,
}

impl<T> ChannelOut<T> {
    fn new(src_id: NodeId, dst_id: NodeId, sender: mpsc::Sender<Option<T>>) -> Self {
        Self {
            _src: src_id,
            _dst: dst_id,
            sender,
        }
    }
}

impl<T: Send> PipeOut<T> for ChannelOut<T> {
    fn dst_id(&self) -> NodeId {
        self._dst
    }

    fn send(&mut self, value: Option<T>) {
        match self.sender.send(value) {
            Ok(_) => {},
            Err(err) => { 
                println!("{:?} Unexpected error {:?}", self._dst, err);
            }
        }
    }
} 

impl<T> fmt::Debug for ChannelOut<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ChannelOut({:?}, {:?})", &self._src, &self._dst)
    }
}

impl<T> fmt::Debug for ChannelIn<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ChannelIn({:?}, {:?})", &self.src_id, &self._dst)
    }
}

pub(crate) trait EagerPipeTrait {
    fn is_pending(&self) -> bool;
    fn src_id(&self) -> NodeId;
    fn src_index(&self) -> usize;
    fn n_read(&self) -> u64;
}

#[derive(Clone)]
pub struct EagerPipe<T: FlowData> {
    pipe: Arc<Mutex<PipeInner<T>>>,
}

impl<T: FlowData> EagerPipe<T> {
    pub(crate) fn new(src_id: SourceId<T>, src_index: usize, dst_id: NodeId, dst_index: usize) -> Self {
        let inner = PipeInner {
            src_id,
            src_index,

            dst_id,
            dst_index,

            value: Out::None,

            n_request: 0, 
            n_sent: 0,
            n_read: 0,
        };

        Self {
            pipe: Arc::new(Mutex::new(inner))
        }
    }
}

impl<T: FlowData> EagerPipeTrait for EagerPipe<T> {
    fn is_pending(&self) -> bool {
        self.pipe.lock().unwrap().value.is_pending()
    }

    fn src_id(&self) -> NodeId {
        self.pipe.lock().unwrap().src_id.id()
    }

    fn src_index(&self) -> usize {
        self.pipe.lock().unwrap().src_index
    }

    fn n_read(&self) -> u64 {
        self.pipe.lock().unwrap().n_read
    }
}

impl<T: FlowData> EagerPipe<T> {
    pub(crate) fn data_request(&self, n_request: u64) -> bool {
        self.pipe.lock().unwrap().data_request(n_request)
    }

    pub(crate) fn send(&self, value: Out<T>) {
        self.pipe.lock().unwrap().send(value);
    }
}

impl<T: FlowData> PipeIn<T> for EagerPipe<T> {
    fn out_index(&self) -> usize {
        self.pipe.lock().unwrap().src_index
    }

    fn init(&mut self) {
        todo!()
    }

    fn fill(&mut self, waker: &mut dyn InnerWaker) -> bool {
        todo!()
    }

    fn n_read(&self) -> u64 {
        self.pipe.lock().unwrap().n_read
    }

    fn next(&mut self) -> Option<T> {
        self.pipe.lock().unwrap().next()
    }
}

pub struct PipeInner<T: FlowData> {
    // id: PipeId,

    src_id: SourceId<T>,
    src_index: usize,

    dst_id: NodeId,
    dst_index: usize,

    value: Out<T>,

    n_request: u64,
    n_sent: u64,
    n_read: u64,
}

impl<T: FlowData> PipeInner<T> {
    fn data_request(&mut self, n_request: u64) -> bool {
        if self.n_read < self.n_request && self.value.is_pending() {
            self.n_request = n_request;

            true
        } else {
            false
        }
    }

    fn next(&mut self) -> Option<T> {
        match self.value.take() {
            Out::None => {
                self.n_read += 1;

                None
            },
            Out::Some(value) => {
                self.n_read += 1;

                self.value.replace(Out::Pending);

                Some(value)
            },
            Out::Pending => todo!(),
        }

    }

    fn send(&mut self, value: Out<T>) {
        assert!(self.value.is_pending());
        assert!(! value.is_pending());

        self.value.replace(value);

        self.n_sent += 1;
    }
}
