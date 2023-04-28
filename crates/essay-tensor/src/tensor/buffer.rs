use std::{ptr::NonNull, alloc::Layout, alloc, ops::{Index, IndexMut}};

use super::tensor::Dtype;

pub struct TensorData<D:Dtype=f32> {
    data: NonNull<D>,
    len: usize,
}

pub struct TensorUninit<D:Dtype=f32> {
    data: NonNull<D>,
    len: usize,
}

impl<D:Dtype> TensorData<D> {
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn get(&self, offset: usize) -> Option<D> {
        if offset < self.len {
            unsafe { Some(self.get_unchecked(offset)) }
        } else {
            None
        }
    }

    #[inline]
    pub unsafe fn get_unchecked(&self, offset: usize) -> D {
        *self.data.as_ptr().add(offset)
    }

    #[inline]
    pub unsafe fn as_ptr(&self) -> *const D {
        self.data.as_ptr()
    }
}

impl<D:Dtype> Index<usize> for TensorData<D> {
    type Output = D;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.len());

        unsafe {
            self.data.as_ptr().add(index).as_ref().unwrap_unchecked()
        }
    }
}

impl<D:Dtype + PartialEq> PartialEq for TensorData<D> {
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }

        for i in 0..self.len {
            if self.get(i) != other.get(i) {
                return false;
            }
        }

        return true;
    }
}

impl<D:Dtype> TensorUninit<D> {
    pub unsafe fn new(len: usize) -> Self {
        let layout = Layout::array::<D>(len).unwrap();
        
        let data =
            NonNull::<D>::new_unchecked(
                alloc::alloc(layout).cast::<D>());
        
        Self {
            data,
            len,
        }
    }

    pub unsafe fn init(self) -> TensorData<D> {
        TensorData {
            data: self.data,
            len: self.len,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub unsafe fn as_ptr(&self) -> *mut D {
        self.data.as_ptr()
    }

    #[inline]
    pub unsafe fn get_unchecked(&self, offset: usize) -> D {
        *self.data.as_ptr().add(offset)
    }

    #[inline]
    pub unsafe fn set_unchecked(&mut self, offset: usize, value: D) {
        *self.data.as_ptr().add(offset) = value;
    }
}

impl<D:Dtype> Index<usize> for TensorUninit<D> {
    type Output = D;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.len());

        unsafe {
            self.data.as_ptr().add(index).as_ref().unwrap_unchecked()
        }
    }
}

impl<D:Dtype> IndexMut<usize> for TensorUninit<D> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(index < self.len());

        unsafe {
            self.data.as_ptr().add(index).as_mut().unwrap_unchecked()
        }
    }
}
