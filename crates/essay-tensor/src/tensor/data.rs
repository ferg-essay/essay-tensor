use core::{slice};
use std::{
    ptr::{NonNull, self}, 
    alloc::Layout, alloc, 
    ops::{Index, self, IndexMut, RangeBounds}, 
    slice::SliceIndex, 
    any::TypeId, 
    marker::PhantomData
};

pub(crate) struct TensorData<T> {
    // type_id: TypeId,
    len: usize,
    // item_size: usize,

    data: NonNull<T>,

    // drop: Option<Box<dyn Fn(&NonNull<u8>)>>,
}

impl<T> TensorData<T> {
    pub fn from_vec(mut vec: Vec<T>) -> Self {
        let len = vec.len();
        let layout = Layout::array::<T>(len).unwrap();

        unsafe {
            let data = NonNull::<T>::new_unchecked(alloc::alloc(layout).cast::<T>());

            let ptr = data.cast::<T>().as_ptr();
            let mut i = 0;
            for value in vec.drain(..) {
                ptr::write(ptr.add(i), value);
                i += 1;
            }

            // Self::new_with_drop::<T>(data, len)
            Self::new(data, len)
        }
    }

    fn new(data: NonNull<T>, len: usize) -> Self {
        Self {
            // type_id: TypeId::of::<T>(),
            len,

            data,
        }
    }

    /*
    fn new_with_drop(data: NonNull<u8>, len: usize) -> Self {
        Self {
            type_id: TypeId::of::<T>(),
            len,

            data,
            drop: Self::box_drop::<T>()
        }
    }

    fn box_drop() -> Option<Box<dyn Fn(&NonNull<u8>)>> {
        Some(Box::new(|data: &NonNull<u8>| {
            unsafe {
                ptr::drop_in_place(data.cast::<T>().as_ptr())
            }
        }))
    }
    */

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn checkcast(&self, len: usize) {
        // assert_eq!(self.type_id, TypeId::of::<T>());
        assert_eq!(self.len, len);
    }

    #[inline(always)]
    pub unsafe fn as_ptr(&self) -> *const T {
        //self.data.cast::<T>().as_ptr()
        self.data.as_ptr()
    }

    // Returns a possibly-wrapped pointer at the offset to support
    // broadcast
    #[inline]
    pub unsafe fn as_wrap_ptr(&self, offset: usize) -> *const T {
        if offset < self.len {
            self.as_ptr().add(offset)
        } else {
            self.as_ptr().add(offset % self.len)
        }
    }

    #[inline]
    pub unsafe fn as_slice(&self) -> &[T] {
        ptr::slice_from_raw_parts(self.as_ptr(), self.len())
            .as_ref()
            .unwrap()
    }

    #[inline]
    pub unsafe fn as_sub_slice(&self, offset: usize, len: usize) -> &[T] {
        assert!(offset <= self.len());
        assert!(offset + len <= self.len());

        ptr::slice_from_raw_parts(self.as_ptr().add(offset), len)
            .as_ref()
            .unwrap()
    }

    #[inline]
    pub unsafe fn as_wrap_slice(&self, range: impl RangeBounds<usize>) -> &[T] {
        use ops::Bound::*;
    
        match (range.start_bound(), range.end_bound()) {
            (Included(offset), Unbounded) => {
                let offset = if *offset < self.len() { 
                    *offset 
                } else { 
                    *offset % self.len() 
                };

                self.as_sub_slice(offset, self.len() - offset)
            }
            (Included(offset), Excluded(tail)) => {
                let offset = if *offset < self.len() { 
                    *offset 
                } else { 
                    *offset % self.len() 
                };

                assert!(offset < *tail);
                assert!(*tail <= self.len());

                self.as_sub_slice(offset, *tail - offset)
            }
            (Unbounded, Unbounded) => self.as_slice(),
            _ => unimplemented!(),
        }
    }

    #[inline]
    pub unsafe fn get(&self, offset: usize) -> Option<&T> {
        if offset < self.len {
            self.as_ptr().add(offset).as_ref()
        } else {
            None
        }
    }
}

impl<T:Clone> TensorData<T> {
    pub(crate) fn from_slice(value: &[T]) -> Self {
        let layout = Layout::array::<T>(value.len()).unwrap();

        unsafe {
            let data = NonNull::<T>::new_unchecked(alloc::alloc(layout).cast::<T>());

            let ptr = data.as_ptr();
            for i in 0..value.len() {
                ptr::write(ptr.add(i), value[i].clone())
            }

            Self::new(data, value.len())            
        }
    }

}

impl<T:Copy> TensorData<T> {
    #[inline]
    pub unsafe fn get_unchecked(&self, offset: usize) -> T {
        *self.as_ptr().add(offset)
    }

    #[inline]
    pub unsafe fn get_wrap(&self, offset: usize) -> Option<T> {
        if offset < self.len {
            Some(self.get_unchecked(offset))
        } else {
            Some(self.get_unchecked(offset % self.len))
        }
    }

    #[inline]
    pub unsafe fn read_wrap(&self, offset: usize) -> T {
        if offset < self.len {
            self.get_unchecked(offset)
        } else {
            self.get_unchecked(offset % self.len)
        }
    }
}
/*
impl<T: Drop> Drop for TensorData<T> {
    fn drop(&mut self) {
        if let Some(drop) = &self.drop {
            (drop)(&self.data)
        }
    }
}
*/

// unsafe: TensorData is read-only
unsafe impl<T: Send> Send for TensorData<T> {}
unsafe impl<T: Sync> Sync for TensorData<T> {}

pub struct TensorUninit<T=f32> {
    type_id: TypeId,
    len: usize,
    // item_size: usize,

    data: NonNull<T>,
}

impl<T:'static + Copy> TensorUninit<T> {
    pub unsafe fn new(len: usize) -> Self {
        let layout = Layout::array::<T>(len).unwrap();
        
        Self {
            type_id: TypeId::of::<T>(),
            len,

            data: NonNull::<T>::new_unchecked(alloc::alloc(layout).cast::<T>()),
        }
    }

    pub(crate) unsafe fn init(self) -> TensorData<T> {
        TensorData::new(self.data, self.len)
        /*
        TensorData {
            type_id: self.type_id,
            len: self.len,

            data: self.data,

            drop: None,
        }
        */
    }
}

impl<T> TensorUninit<T> {
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub unsafe fn as_ptr(&self) -> *mut T {
        self.data.cast::<T>().as_ptr()
    }

    #[inline]
    pub unsafe fn as_mut_ptr(&mut self) -> *mut T {
        self.data.cast::<T>().as_mut()
    }

    #[inline]
    pub unsafe fn as_mut(&mut self) -> &mut T {
        self.data.cast::<T>().as_mut()
    }

    #[inline]
    pub unsafe fn set_unchecked(&mut self, offset: usize, value: T) {
        *self.data.cast::<T>().as_ptr().add(offset) = value;
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
    pub fn as_mut_slice(&mut self) -> &mut [T] {
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
        *self.as_ptr().add(offset)
    }
}

impl<T> ops::Deref for TensorUninit<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.as_ptr(), self.len) }
    }
}

impl<T, I: SliceIndex<[T]>> Index<I> for TensorUninit<T> {
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        Index::index(self.as_slice(), index)
    }
}

impl<T, I: SliceIndex<[T]>> IndexMut<I> for TensorUninit<T> {
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        IndexMut::index_mut(self.as_mut_slice(), index)
    }
}

#[cfg(test)]
mod test {
    use std::{rc::Rc, cell::RefCell};

    use crate::{prelude::*, tensor::{Dtype, tensor::Shape}};

    #[test]
    fn test_drop() {
        let ptr = {
            let test = Test::new(2);
            let ptr = test.ptr.clone();
            let _tensor = Tensor::from(test);

            ptr
        };

        assert_eq!(take(&ptr), "Drop[2], Drop[2]");

        let ptr = {
            let vec = vec![Test::new(2)];
            let ptr = vec[0].ptr.clone();
            let _tensor = Tensor::from_vec(vec, Shape::from(1));

            ptr
        };

        assert_eq!(take(&ptr), "Drop[2]");
    }

    #[test]
    fn test_drop_clone() {
        let ptr = {
            let vec = vec![Test::new(2)];
            let ptr = vec[0].ptr.clone();
            let _tensor = Tensor::from_vec(vec, Shape::from(1));
            let _tensor2 = _tensor.clone();

            ptr
        };

        assert_eq!(take(&ptr), "Drop[2]");
    }

    fn take(ptr: &Rc<RefCell<Vec<String>>>) -> String {
        let vec : Vec<String> = ptr.borrow_mut().drain(..).collect();

        vec.join(", ")
    }

    #[derive(Debug, Clone)]
    struct Test {
        id: usize,
        ptr: Rc<RefCell<Vec<String>>>,
    }

    impl Test {
        fn new(id: usize) -> Self {
            Self {
                id,
                ptr: Rc::new(RefCell::new(Vec::default())),
            }
        }
    }

    impl Dtype for Test {}

    impl Drop for Test {
        fn drop(&mut self) {
            self.ptr.borrow_mut().push(format!("Drop[{:?}]", self.id));
        }
    }

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

    /*
    #[test]
    fn test_slice_index() {
        let a = tensor!(10.);
        assert_eq!(&a.data()[..], &[10.]);
        assert_eq!(&a.data()[1..], &[]);
    }

    #[test]
    fn test_wrap_slice_index() {
        unsafe {
            let a = tensor!(10.);
            assert_eq!(&a.data().as_wrap_slice(..), &[10.]);
            assert_eq!(&a.data().as_wrap_slice(1..), &[10.]);
            assert_eq!(&a.data().as_wrap_slice(2..), &[10.]);

            let a = tensor!([10., 20.]);
            assert_eq!(&a.data().as_wrap_slice(..), &[10., 20.]);
            assert_eq!(&a.data().as_wrap_slice(1..), &[20.]);
            assert_eq!(&a.data().as_wrap_slice(2..), &[10., 20.]);
        }
    }
    */

    /*
    #[test]
    fn test_slice_mut() {
        unsafe {
            let mut a = TensorUninit::<f32>::new(1);
            let slice = a.as_mut_slice();
            slice[0] = 10.0;
            assert_eq!(slice, &[10.]);
            let a = a.init();

            assert_eq!(a.as_slice(), &[10.]);
        };
    }
    */
}