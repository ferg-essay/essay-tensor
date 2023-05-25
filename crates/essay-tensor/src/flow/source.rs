use core::fmt;
use std::{marker::PhantomData, sync::{Arc, Mutex}, mem};

use super::{FlowIn, FlowData, In};


pub trait Source<I, O> : Send + 'static
where
    I: FlowIn<I> + 'static,
    O: 'static,
{
    fn init(&mut self) {}
    
    fn next(&mut self, input: &mut I::Input) -> Result<Out<O>>;
}

pub trait SourceFactory<I, O> : Send + 'static
where
    I: FlowIn<I> + 'static,
    O: 'static,
{
    fn new(&mut self) -> Box<dyn Source<I, O>>;
}

#[derive(Copy, PartialEq)]
pub struct SourceId<T> {
    index: usize,
    marker: PhantomData<T>,
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct NodeId(usize);

#[derive(Debug)]
pub struct SourceErr;

pub type Result<T> = std::result::Result<T, SourceErr>;

impl<T> fmt::Debug for SourceId<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TaskId[{}]", self.index)
    }
}

impl<T> Clone for SourceId<T> {
    fn clone(&self) -> Self {
        Self { 
            index: self.index,
            marker: self.marker.clone() 
        }
    }
}

impl NodeId {
    pub fn index(&self) -> usize {
        self.0
    }
}

impl<T: 'static> SourceId<T> {
    pub(crate) fn new(index: usize) -> Self {
        Self {
            index,
            marker: PhantomData,
        }
    }

    #[inline]
    pub fn id(&self) -> NodeId {
        NodeId(self.index)
    }

    #[inline]
    pub fn index(&self) -> usize {
        self.index
    }
}


//
// Output task
//
#[derive(Clone)]
pub struct OutputData<T: Clone + Send + 'static> {
    marker: PhantomData<T>,
}

impl<T: Clone + Send + Sync + 'static> FlowData for OutputData<T> {}

pub struct OutputSource<O: FlowIn<O>> {
    data: SharedOutput<O>,
}

impl<O: FlowIn<O>> OutputSource<O> {
    pub(crate) fn new(data: SharedOutput<O>) -> Self {
        Self {
            data,
        }
    }

    fn data(&self) -> &SharedOutput<O> {
        &self.data
    }
}

impl<O: FlowIn<O>> Source<O, bool> for OutputSource<O> {
    fn next(&mut self, input: &mut O::Input) -> Result<Out<bool>> {
        let value = O::next(input);

        match value {
            Out::Some(value) => {
                self.data.replace(value);
                Ok(Out::Some(true))
            },
            Out::None => {
                self.data.take();
                Ok(Out::None)
            },
            Out::Pending => {
                Ok(Out::Pending)
            }
        }
    }
}

pub enum Out<T> {
    None,
    Some(T),
    Pending,
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

impl<T> From<Option<T>> for Out<T> {
    fn from(value: Option<T>) -> Self {
        match value {
            Some(value) => Out::Some(value),
            None => Out::None,
        }
    }
}

#[derive(Clone)]
pub struct SharedOutput<O> {
    value: Arc<Mutex<Option<O>>>,
}

impl<O> SharedOutput<O> {
    pub fn new() -> Self {
        Self {
            value: Arc::new(Mutex::new(None)),
        }
    }

    pub(crate) fn take(&self) -> Option<O> {
        self.value.lock().unwrap().take()
    }

    pub(crate) fn replace(&self, value: O) {
        self.value.lock().unwrap().replace(value);
    }
}

pub struct TailSource;

impl Source<bool, bool> for TailSource {
    fn next(&mut self, source: &mut In<bool>) -> Result<Out<bool>> {
        source.next(); // assert!(if let Some(true) = source.next() { true } else { false });

        Ok(Out::Some(true))
    }
}

pub struct UnsetSource<I: FlowIn<I>, O: FlowIn<O>> {
    marker: PhantomData<(I, O)>,
}

impl<I: FlowIn<I>, O: FlowIn<O>> UnsetSource<I, O> {
    pub(crate) fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

impl<I: FlowIn<I>, O: FlowIn<O>> Source<I, O> for UnsetSource<I, O> {
    fn next(&mut self, source: &mut I::Input) -> Result<Out<O>> {
        panic!();
    }
}

//
// Function Source
//

impl<I, O, F> Source<I, O> for F
where
    I: FlowIn<I> + 'static,
    O: 'static,
    F: FnMut(&mut I::Input) -> Result<Out<O>> + Send + 'static
{
    fn next(&mut self, input: &mut I::Input) -> Result<Out<O>> {
        self(input)
    }

    fn init(&mut self) {}
}

impl<I, O, F: FnMut() -> S, S> SourceFactory<I, O> for F
where
    I: FlowIn<I> + 'static,
    O: FlowData,
    F: Send + 'static,
    S: Source<I, O>
{
    fn new(&mut self) -> Box<dyn Source<I, O>> {
        Box::new(self())
    }
}
