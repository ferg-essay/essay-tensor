use core::fmt;
use std::{mem, sync::mpsc::{self}};

use super::{data::FlowData, dispatch::{InnerWaker}, source::{NodeId, SourceId}};


pub enum Out<T> {
    None,
    Some(T),
    Pending,
}

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
                panic!("PipeIn.next() can't be called twice in a task");
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
        self.sender.send(value).unwrap();
    }
} 

impl<T> Out<T> {
    #[inline]
    pub fn is_none(&self) -> bool {
        match self {
            Out::None => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_pending(&self) -> bool {
        match self {
            Out::Pending => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_some(&self) -> bool {
        match self {
            Out::Some(_) => true,
            _ => false,
        }
    }

    #[inline]
    pub fn take(&mut self) -> Self {
        match self {
            Out::None => Out::None,
            Out::Some(_) => mem::replace(self, Out::Pending),
            Out::Pending => Out::Pending,
        }
    }

    #[inline]
    pub fn replace(&mut self, value: Self) -> Self {
        mem::replace(self, value)
    }

    #[inline]
    pub fn unwrap(&mut self) -> T {
        if let Out::Some(_) = self {
            let v = mem::replace(self, Out::Pending);
            if let Out::Some(v) = v {
                return v
            }
        }

        panic!("Unwrap with invalid value")
    }
}

impl<T> Default for Out<T> {
    fn default() -> Self {
        Out::None
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
