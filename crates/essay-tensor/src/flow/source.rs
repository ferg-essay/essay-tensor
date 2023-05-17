use std::mem;


pub enum Out<T> {
    None,
    Some(T),
    Pending,
}

pub struct Source<T> {
    item: Out<T>,
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

impl<T> Default for Source<T> {
    fn default() -> Self {
        Self { 
            item: Default::default() 
        }
    }
}

impl<T> Source<T> {
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
