use core::{slice, fmt};
use std::{ptr::{NonNull, self}, alloc::Layout, alloc, 
    ops::{Index, self, IndexMut, RangeBounds}, 
    slice::SliceIndex
};

use super::tensor::Dtype;

pub struct TensorData<T=f32> {
    data: NonNull<T>,
    len: usize,
}

pub struct TensorUninit<T=f32> {
    data: NonNull<T>,
    len: usize,
}

impl<T> TensorData<T> {
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub unsafe fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    // Returns a possibly-wrapped pointer at the offset to support
    // broadcast
    #[inline]
    pub unsafe fn as_wrap_ptr(&self, offset: usize) -> *const T {
        if offset < self.len {
            self.data.as_ptr().add(offset)
        } else {
            self.data.as_ptr().add(offset % self.len)
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            ptr::slice_from_raw_parts(self.as_ptr(), self.len())
                .as_ref()
                .unwrap()
        }
    }

    #[inline]
    pub fn as_wrap_slice(&self, range: impl RangeBounds<usize>) -> &[T] {
        use ops::Bound::*;
    
        match (range.start_bound(), range.end_bound()) {
            (Included(offset), Unbounded) => {
                let offset = if *offset < self.len() { 
                    *offset 
                } else { 
                    *offset % self.len() 
                };

                unsafe {
                    ptr::slice_from_raw_parts(
                        self.as_ptr().add(offset), self.len() - offset)
                        .as_ref()
                        .unwrap()
                }
            }
            (Unbounded, Unbounded) => self.as_slice(),
            _ => unimplemented!(),
        }
    }

    #[inline]
    pub fn get(&self, offset: usize) -> Option<&T> {
        if offset < self.len {
            unsafe { self.data.as_ptr().add(offset).as_ref() }
        } else {
            None
        }
    }
}

impl<D:Copy> TensorData<D> {
    #[inline]
    pub unsafe fn get_unchecked(&self, offset: usize) -> D {
        *self.data.as_ptr().add(offset)
    }

    #[inline]
    pub fn get_wrap(&self, offset: usize) -> Option<D> {
        unsafe {
           if offset < self.len {
                Some(self.get_unchecked(offset))
            } else {
                Some(self.get_unchecked(offset % self.len))
            }
        }
    }

    #[inline]
    pub fn read_wrap(&self, offset: usize) -> D {
        unsafe {
           if offset < self.len {
                self.get_unchecked(offset)
            } else {
                self.get_unchecked(offset % self.len)
            }
        }
    }
}

impl<D:Dtype, I: SliceIndex<[D]>> Index<I> for TensorData<D> {
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        Index::index(self.as_slice(), index)
    }
}
/*
impl<D:Dtype> Index<usize> for TensorData<D> {
    type Output = D;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let index = if index < self.len() { index } else { index % self.len() };

        unsafe {
            self.data.as_ptr().add(index).as_ref().unwrap_unchecked()
        }
    }
}
*/

impl<D:PartialEq + Copy> PartialEq for TensorData<D> {
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

impl<D:fmt::Debug + Copy> fmt::Debug for TensorData<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TensorData[")?;

        for i in 0..self.len() {
            if i != 0 {
                write!(f, ", ")?;
            }

            write!(f, "{:?}", self.get(i).unwrap())?;
        }

        write!(f, "]")
    }
}

impl<D:Dtype> ops::Deref for TensorData<D> {
    type Target = [D];

    #[inline]
    fn deref(&self) -> &[D] {
        unsafe { slice::from_raw_parts(self.as_ptr(), self.len) }
    }
}

unsafe impl<D:Dtype + Sync> Sync for TensorData<D> {}
unsafe impl<D:Dtype + Send> Send for TensorData<D> {}

impl<T> TensorUninit<T> {
    pub unsafe fn new(len: usize) -> Self {
        let layout = Layout::array::<T>(len).unwrap();
        
        let data =
            NonNull::<T>::new_unchecked(
                alloc::alloc(layout).cast::<T>());
        
        Self {
            data,
            len,
        }
    }

    pub unsafe fn init(self) -> TensorData<T> {
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
    pub unsafe fn as_ptr(&self) -> *mut T {
        self.data.as_ptr()
    }

    #[inline]
    pub unsafe fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut()
    }

    #[inline]
    pub unsafe fn as_mut(&mut self) -> &mut T {
        self.data.as_mut()
    }

    #[inline]
    pub unsafe fn set_unchecked(&mut self, offset: usize, value: T) {
        *self.data.as_ptr().add(offset) = value;
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            ptr::slice_from_raw_parts(self.as_ptr(), self.len())
                .as_ref()
                .unwrap()
        }
    }

    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        unsafe {
            ptr::slice_from_raw_parts_mut(self.as_mut_ptr(), self.len())
                .as_mut()
                .unwrap()
        }
    }
}

impl<T:Copy> TensorUninit<T> {
    #[inline]
    pub unsafe fn get_unchecked(&self, offset: usize) -> T {
        *self.data.as_ptr().add(offset)
    }
}

impl<D:Dtype> ops::Deref for TensorUninit<D> {
    type Target = [D];

    #[inline]
    fn deref(&self) -> &[D] {
        unsafe { slice::from_raw_parts(self.as_ptr(), self.len) }
    }
}
/*
impl<D:Dtype> ops::DerefMut for TensorUninit<D> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [D] {
        unsafe { slice::from_raw_parts_mut(self.as_ptr(), self.len) }
    }
}

impl<D:Dtype, I: SliceIndex<[D]>> Index<I> for TensorUninit<D> {
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        Index::index(self.as_slice(), index)
    }
}
*/

impl<D, I: SliceIndex<[D]>> Index<I> for TensorUninit<D> {
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        Index::index(self.as_slice(), index)
    }
}

impl<D, I: SliceIndex<[D]>> IndexMut<I> for TensorUninit<D> {
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        IndexMut::index_mut(self.as_slice_mut(), index)
    }
}

#[cfg(test)]
mod test {
    use crate::{prelude::*, tensor::TensorUninit};

    #[test]
    fn test_slice() {
        let a = tensor!(10.);
        assert_eq!(a.as_slice(), &[10.]);

        let a = tensor!([10.]);
        assert_eq!(a.as_slice(), &[10.]);

        let a = tensor!([10., 20., 30.]);
        assert_eq!(a.as_slice(), &[10., 20., 30.]);

        let a = tensor!([[10., 20.], [30., 40.]]);
        assert_eq!(a.as_slice(), &[10., 20., 30., 40.]);
    }

    #[test]
    fn test_slice_index() {
        let a = tensor!(10.);
        assert_eq!(&a.data()[..], &[10.]);
        assert_eq!(&a.data()[1..], &[]);
    }

    #[test]
    fn test_wrap_slice_index() {
        let a = tensor!(10.);
        assert_eq!(&a.data().as_wrap_slice(..), &[10.]);
        assert_eq!(&a.data().as_wrap_slice(1..), &[10.]);
        assert_eq!(&a.data().as_wrap_slice(2..), &[10.]);

        let a = tensor!([10., 20.]);
        assert_eq!(&a.data().as_wrap_slice(..), &[10., 20.]);
        assert_eq!(&a.data().as_wrap_slice(1..), &[20.]);
        assert_eq!(&a.data().as_wrap_slice(2..), &[10., 20.]);
    }

    #[test]
    fn test_slice_mut() {
        let a = unsafe {
            let mut a = TensorUninit::<f32>::new(1);
            let slice = a.as_slice_mut();
            slice[0] = 10.0;
            assert_eq!(slice, &[10.]);
            a.init()
        };
        assert_eq!(a.as_slice(), &[10.]);
    }
}