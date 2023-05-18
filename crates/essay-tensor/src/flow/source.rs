use core::fmt;
use std::{mem, sync::mpsc::{self}};

use super::{graph::{NodeId, TaskId}, data::FlowData};


pub enum Out<T> {
    None,
    Some(T),
    Pending,
}

pub struct SourceImpl<T> {
    item: Out<T>,
}

pub enum SourceErr {
    Pending
}

pub trait SourceTrait<T> {
    fn is_available(&mut self) -> bool;

    fn next(&mut self) -> Option<T>;
}

pub trait SinkTrait<T> {
    fn send(&mut self, value: Option<T>);
}

pub struct ChannelSink<T> {
    _src: NodeId,
    _dst: NodeId,

    sender: mpsc::Sender<Option<T>>,
}

pub struct ChannelSource<T> {
    _src: NodeId,
    _dst: NodeId,

    receiver: mpsc::Receiver<Option<T>>,
    value: Out<T>,
}

pub fn task_channel<T: 'static>(
    src_id: TaskId<T>, 
    dst_id: NodeId
) -> (Box<dyn SourceTrait<T>>, Box<dyn SinkTrait<T>>) {
    let (sender, receiver) = mpsc::channel::<Option<T>>();

    (
        Box::new(ChannelSource::new(src_id.id(), dst_id, receiver)),
        Box::new(ChannelSink::new(src_id.id(), dst_id, sender))
    )
}

impl<T> ChannelSource<T> {
    fn new(src_id: NodeId, dst_id: NodeId, receiver: mpsc::Receiver<Option<T>>) -> Self {
        Self {
            _src: src_id,
            _dst: dst_id,
            receiver,
            value: Out::None,
        }
    }
}

impl<T> ChannelSink<T> {
    fn new(src_id: NodeId, dst_id: NodeId, sender: mpsc::Sender<Option<T>>) -> Self {
        Self {
            _src: src_id,
            _dst: dst_id,
            sender,
        }
    }
}

impl<T> SinkTrait<T> for ChannelSink<T> {
    fn send(&mut self, value: Option<T>) {
        self.sender.send(value).unwrap();
    }
} 

pub struct Source<T>(Box<dyn SourceTrait<T>>);

impl<T> Source<T> {
    pub fn next(&mut self) -> Option<T> {
        self.0.next()
    }
}

impl<T> Source<T>
where
    T: FlowData + Clone + 'static
{
    pub(crate) fn new(dst_id: Box<dyn SourceTrait<T>>) -> Source<T>
    {
        todo!()
    }
}

impl<T> SourceTrait<T> for ChannelSource<T> {
    fn next(&mut self) -> Option<T> {
        match self.value.take() {
            Out::None => None,
            Out::Some(value) => Some(value),
            Out::Pending => {
                panic!("Source.next() can't be called twice in a task");
            }
        }
    }

    fn is_available(&mut self) -> bool {
        if ! self.value.is_pending() {
            true
        } else {
            match self.receiver.try_recv() {
                Ok(Some(value)) => { 
                    self.value.replace(Out::Some(value)); 
                    true 
                }

                Ok(None) => { 
                    self.value.replace(Out::None);
                    true 
                }

                Err(err) => {
                    match err {
                        mpsc::TryRecvError::Empty => false,
                        mpsc::TryRecvError::Disconnected => todo!(),
                    }
                }
            }
        }
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

impl<T> fmt::Debug for ChannelSink<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ChannelSink({:?}, {:?})", &self._src, &self._dst)
    }
}

impl<T> fmt::Debug for ChannelSource<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ChannelSource({:?}, {:?})", &self._src, &self._dst)
    }
}

impl<T> Default for SourceImpl<T> {
    fn default() -> Self {
        Self { 
            item: Default::default() 
        }
    }
}

impl<T> SourceImpl<T> {
    pub(crate) fn new() -> Self {
        Self {
            item: Out::Pending,
        }
    }

    pub fn next(&mut self) -> Out<T> {
        self.item.take()        
    }

    pub fn push(&mut self, value: T) {
        assert!(self.item.is_none());

        self.item.replace(Out::Some(value));
    }

    #[inline]
    pub(crate) fn is_some(&self) -> bool {
        self.item.is_some()
    }

    #[inline]
    pub(crate) fn is_none(&self) -> bool {
        self.item.is_none()
    }
}
