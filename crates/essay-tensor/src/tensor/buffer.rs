use std::{ptr::NonNull, alloc::Layout, alloc};

use super::tensor::Dtype;


pub struct TensorData<D:Dtype=f32> {
    data: NonNull<D>,
    len: usize,
}

impl<D:Dtype> TensorData<D> {
    pub unsafe fn new_uninit(len: usize) -> Self {
        let layout = Layout::array::<D>(len).unwrap();
        
        let data =
            NonNull::<D>::new_unchecked(
                alloc::alloc(layout).cast::<D>());
        
        Self {
            data,
            len,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn get(&self, offset: usize) -> Option<D> {
        if offset < self.len {
            unsafe { Some(self.uget(offset)) }
        } else {
            None
        }
    }

    #[inline]
    pub unsafe fn uget(&self, offset: usize) -> D {
        *self.data.as_ptr().add(offset)
    }

    #[inline]
    pub unsafe fn ptr(&self) -> *mut D {
        self.data.as_ptr()
    }

    #[inline]
    pub unsafe fn set(&mut self, offset: usize, value: D) {
        assert!(offset < self.len);

        self.uset(offset, value);
    }

    #[inline]
    pub unsafe fn uset(&mut self, offset: usize, value: D) {
        *self.data.as_ptr().add(offset) = value;
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