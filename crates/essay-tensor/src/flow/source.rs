use core::fmt;
use std::{marker::PhantomData, sync::{Arc, Mutex}, mem, collections::VecDeque};

use super::{FlowIn, FlowData, In, flow};


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

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd)]
pub struct NodeId(usize);

#[derive(Debug)]
pub struct SourceErr;

pub type Result<T> = std::result::Result<T, SourceErr>;

impl<T> fmt::Debug for SourceId<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("SourceId").field(&self.index).finish()
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

impl<T> From<&NodeId> for SourceId<T> {
    fn from(value: &NodeId) -> Self {
        Self {
            index: value.index(),
            marker: PhantomData,
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
#[derive(Clone, Debug)]
pub struct OutputData<T: Clone + Send + 'static> {
    marker: PhantomData<T>,
}

impl<T: Clone + Send + Sync + fmt::Debug + 'static> FlowData for OutputData<T> {}

pub struct OutputSource<O: FlowIn<O>> {
    data: SharedOutput<O>,
}

impl<O: FlowIn<O>> OutputSource<O> {
    pub(crate) fn new(data: SharedOutput<O>) -> Self {
        Self {
            data,
        }
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
            Out::Some(_) => mem::replace(self, Out::<T>::Pending),
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
            let v = mem::replace(self, Out::<T>::Pending);
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

impl<T: fmt::Debug> fmt::Debug for Out<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::Some(arg0) => f.debug_tuple("Some").field(arg0).finish(),
            Self::Pending => write!(f, "Pending"),
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

pub struct NoneSourceFactory<I: FlowIn<I>, O: FlowIn<O>> {
    marker: PhantomData<(I, O)>,
}

impl<I: FlowIn<I>, O: FlowIn<O>> NoneSourceFactory<I, O> {
    pub(crate) fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

impl<I: FlowIn<I>, O: FlowIn<O>> SourceFactory<I, O> for NoneSourceFactory<I, O> {
    fn new(&mut self) -> Box<dyn Source<I, O>> {
        todo!()
    }
}

pub struct NoneSource<I: FlowIn<I>, O: FlowIn<O>> {
    marker: PhantomData<(I, O)>,
}

impl<I: FlowIn<I>, O: FlowIn<O>> NoneSource<I, O> {
    pub(crate) fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

impl<I: FlowIn<I>, O: FlowIn<O>> Source<I, O> for NoneSource<I, O> {
    fn next(&mut self, _source: &mut I::Input) -> Result<Out<O>> {
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

pub struct VecSource<T: FlowData> {
    vec: Vec<T>,
    index: usize,
}

impl<T: FlowData + Clone> Source<(), T> for VecSource<T> {
    fn next(&mut self, input: &mut ()) -> Result<Out<T>> {
        let index = self.index;
        self.index += 1;
        println!("Vec[{index}]: {:?} ", &self.vec);
        match self.vec.get(index) {
            Some(value) => Ok(Out::Some(value.clone())),
            None => Ok(Out::None),
        }
    }
}

impl<T: FlowData> From<Vec<T>> for VecSource<T> {
    fn from(value: Vec<T>) -> Self {
        VecSource {
            vec: value,
            index: 0,
        }
    }
}

impl<T: FlowData> Source<(), T> for VecDeque<T> {
    fn next(&mut self, _input: &mut ()) -> Result<Out<T>> {
        Ok(Out::from(self.pop_front()))
    }
}
