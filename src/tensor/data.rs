use core::slice;
use std::{
    alloc::{self, Layout}, mem, num::NonZeroUsize, ptr::NonNull
};

use crate::tensor::Tensor;

use super::{Shape, Type};

pub(super) struct TensorData<T: Type> {
    data: NonNull<T>,
    len: NonZeroUsize,
}

impl<T: Type> TensorData<T> {
    #[inline]
    fn new(len: NonZeroUsize, data: NonNull<T>, shape: Shape) -> Tensor<T> {
        Tensor::new(Self { len, data }, shape)
    }

    #[inline]
    pub(super) fn from_boxed_slice(slice: Box<[T]>, shape: impl Into<Shape>) -> Tensor<T> {
        let len = NonZeroUsize::new(slice.len())
            .expect("TensorData requires a non-empty slice");

        let shape = shape.into();

        unsafe {
            let ptr = Box::into_raw(slice);
            let data = NonNull::<T>::new_unchecked(ptr as *mut T);

            Self::new(len, data, shape)
        }
    }

    #[inline]
    pub(super) fn from_boxed_rows<const N: usize>(
        slice: Box<[[T; N]]>,
        shape: impl Into<Shape>
    ) -> Tensor<T> {
        let len = NonZeroUsize::new(slice.len() * N).unwrap();
        let shape = shape.into();

        unsafe {
            let ptr = Box::into_raw(slice);
            let data = NonNull::<T>::new_unchecked(ptr as *mut T);

            Self::new(len, data, shape)
        }
    }

    /// Returns the flattened length of the tensor's data.
    #[inline]
    pub(super) fn len(&self) -> usize {
        self.len.get()
    }

    #[inline(always)]
    pub(super) unsafe fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }
}

impl<T: Type> Drop for TensorData<T> {
    fn drop(&mut self) {
        unsafe {
            let len = self.len.get();
            let slice = slice::from_raw_parts_mut(self.data.as_ptr(), len); 
            drop(Box::from_raw(slice));
        }
    }
}

// unsafe: TensorData is read-only
unsafe impl<T: Type + Send> Send for TensorData<T> {}
unsafe impl<T: Type + Sync> Sync for TensorData<T> {}

    
/// This initialization is unsafe because the data is uninitialized,
/// which means the caller must take take to both assign every item,
/// and to only use o.write(data) to update values because other methods 
/// will try to drop the uninitialized previous
pub unsafe fn unsafe_init<T: Type>(
    len: usize, 
    shape: impl Into<Shape>, 
    mut init: impl FnMut(*mut T)
) -> Tensor<T>
{
    let len = NonZeroUsize::new(len)
        .expect("unsafe_init must have a non-zero length");

    let shape = shape.into();

    let layout = Layout::array::<T>(len.get()).unwrap();
    let data = NonNull::<T>::new_unchecked(alloc::alloc(layout).cast::<T>());

    (init)(data.cast::<T>().as_mut());

    let data = mem::ManuallyDrop::new(data);

    TensorData::<T>::new(len, *data, shape)
}
