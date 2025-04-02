use core::slice;
use std::{
    alloc::{self, Layout}, mem, ptr::{self, NonNull}, sync::Arc
};

use crate::Tensor;

use super::{Shape, TensorId};

pub(crate) struct TensorData<T> {
    data: NonNull<T>,
    len: usize,
}

impl<T> TensorData<T> {
    #[inline]
    fn new(data: NonNull<T>, len: usize) -> Self {
        Self {
            data,
            len,
        }
    }

    #[inline]
    pub(crate) fn from_boxed_slice(slice: Box<[T]>) -> Self {
        let len = slice.len();

        unsafe {
            let ptr = Box::into_raw(slice);
            let data = NonNull::<T>::new_unchecked(ptr as *mut T);

            Self {
                data,
                len,
            }
        }
    }

    #[inline]
    pub(crate) fn from_vec(vec: Vec<T>) -> Self {
        Self::from_boxed_slice(Vec::into_boxed_slice(vec))
    }

    #[inline]
    pub(crate) fn from_boxed_rows<const N: usize>(slice: Box<[[T; N]]>) -> TensorData<T> {
        let len = slice.len();
        let size = len * N;

        unsafe {
            let ptr = Box::into_raw(slice);
            let data = NonNull::<T>::new_unchecked(ptr as *mut T);

            TensorData {
                data,
                len: size,
            }
        }
    }

    #[inline]
    pub(crate) fn from_vec_rows<const N: usize>(vec: Vec<[T; N]>) -> TensorData<T> {
        Self::from_boxed_rows(Vec::into_boxed_slice(vec))
    }

    #[inline]
    pub(crate) fn _from_boxed_matrices<const M: usize, const N: usize>(
        slice: Box<[[[T; N]; M]]>
    ) -> TensorData<T>
    {
        let len = slice.len();
        let size = len * N * M;

        unsafe {
            let ptr = Box::into_raw(slice);
            let data = NonNull::<T>::new_unchecked(ptr as *mut T);

            TensorData {
                data,
                len: size,
            }
        }
    }

    #[inline]
    #[cfg(test)]
    pub(crate) fn from_vec_matrices<const M: usize, const N: usize>(
        vec: Vec<[[T; N]; M]>
    ) -> TensorData<T> {
        Self::_from_boxed_matrices(Vec::into_boxed_slice(vec))
    }

    /// Returns the flattened length of the tensor's data.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub unsafe fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
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
    pub unsafe fn get(&self, offset: usize) -> Option<&T> {
        if offset < self.len {
            self.as_ptr().add(offset).as_ref()
        } else {
            None
        }
    }
}

impl<T: Clone> TensorData<T> {
    pub(crate) fn from_slice(value: &[T]) -> Self {
        unsafe {
            let layout = Layout::array::<T>(value.len()).unwrap();
            let data = NonNull::<T>::new_unchecked(alloc::alloc(layout).cast::<T>());

            let ptr = data.as_ptr();
            for (i, value) in value.iter().enumerate() {
                ptr::write(ptr.add(i), value.clone())
            }

            Self::new(data, value.len())            
        }
    }
}

pub(super) struct TensorDataSlice<'a, T> {
    data: &'a Arc<TensorData<T>>,
    offset: usize,
    len: usize,
}

impl<'a, T> TensorDataSlice<'a, T> {
    pub(super) fn new(data: &'a Arc<TensorData<T>>, offset: usize, len: usize) -> Self {
        assert!(offset + len <= data.len());

        Self {
            data,
            offset,
            len
        }
    }

    #[inline]
    pub(super) fn _as_slice(&self) -> &[T] {
        unsafe { self.data.as_sub_slice(self.offset, self.len) }
    }

    // Returns a possibly-wrapped pointer at the offset to support
    // broadcast
    #[inline]
    pub(super) unsafe fn as_wrap_slice(&self, offset: usize) -> &[T] {
        let offset = if offset < self.len {
            offset
        } else {
            offset % self.len
        };

        self.data.as_sub_slice(self.offset + offset, self.len - offset)
    }

    #[inline]
    pub unsafe fn as_ptr(&self) -> *const T {
        self.data.as_ptr().add(self.offset)
    }

    // Returns a possibly-wrapped pointer at the offset to support
    // broadcast
    #[inline]
    pub unsafe fn _as_wrap_ptr(&self, offset: usize) -> *const T {
        if offset < self.len {
            self.data.as_ptr().add(self.offset + offset)
        } else {
            self.data.as_ptr().add(self.offset + offset % self.len)
        }
    }

    pub(super) fn map<U: Clone + 'static>(
        &self, 
        mut f: impl FnMut(&T) -> U
    ) -> TensorData<U> {
        let len = self.len;
        
        unsafe {
            let mut out = TensorUninit::<U>::new(len);
        
            let a_ptr = self.as_ptr();
            let o_ptr = out.as_mut_ptr();
            
            for i in 0..len {
                *o_ptr.add(i) = (f)(a_ptr.add(i).as_ref().unwrap());
            }
        
            out.into()
        }
    }

    pub(crate) fn map2<U, F, V: Clone + 'static>(
        &self, 
        rhs: &TensorDataSlice<U>,
        mut f: F
    ) -> TensorData<V>
    where
        F: FnMut(&T, &U) -> V
    {
        let a_len = self.len;
        let b_len = rhs.len;

        let size = a_len.max(b_len);
        let inner = a_len.min(b_len);
        let batch = size / inner;

        assert!(batch * inner == size, "broadcast mismatch a.len={} b.len={}", a_len, b_len);
        
        unsafe {
            let mut out = TensorUninit::<V>::new(size);

            for n in 0..batch {
                let a = self.as_wrap_slice(n * inner);
                let b = rhs.as_wrap_slice(n * inner);

                let o = out.as_sub_slice(n * inner, inner);

                for k in 0..inner {
                    o[k] = f(&a[k], &b[k]);
                }
            }

            out.into()
        }
    }

    pub(crate) fn map_slice<const M: usize, U: Clone + 'static>(
        &self, 
        n: usize,
        f: impl Fn(&[T]) -> [U; M]
    ) -> TensorData<U> {
        assert!(n > 0);
        assert!(self.len % n == 0);

        let len = self.len / n;

        unsafe {
            let mut out = TensorUninit::<U>::new(M * len);

            let xy = self.as_ptr();
            let o = out.as_mut_slice();

            for i in 0..len {
                let offset = n * i;

                let slice = ptr::slice_from_raw_parts(xy.add(offset), n)
                    .as_ref()
                    .unwrap();

                let value = (f)(slice);

                for j in 0..M {
                    o[offset + j] = value[j].clone();
                }
            }

            out.into()
        }
    }

    pub(super) fn fold<S: Clone + 'static>(
        &self, 
        cols: usize,
        init: S,
        mut f: impl FnMut(S, &T) -> S
    ) -> TensorData<S> {
        let len = self.len / cols;
        assert!(cols * len == self.len);
        
        unsafe {
            let mut out = TensorUninit::<S>::new(len);
        
            let o_ptr = out.as_mut_ptr();
            let mut a_ptr = self.as_ptr();
            
            for i in 0..len {
                let mut value = init.clone();

                for i in 0..cols {
                    value = (f)(value, a_ptr.add(i).as_ref().unwrap());
                }

                *o_ptr.add(i) = value;

                a_ptr = a_ptr.add(cols);
            }
        
            out.into()
        }
    }
}

impl<T: Clone + 'static> TensorData<T> {
    /*
    pub fn map_slice<const N: usize, const M: usize>(
        &self, 
        f: impl Fn(&[T]) -> [T; M]
    ) -> Self {
        assert!(self.len() % N == 0);

        let n = self.len();

        unsafe {
            let mut out = TensorUninit::<T>::new(N * n);

            let xy = self.as_ptr();
            let o = out.as_mut_slice();

            for i in 0..n {
                let offset = N * i;

                let slice = ptr::slice_from_raw_parts(xy.add(offset), N)
                    .as_ref()
                    .unwrap();

                let value = (f)(slice);

                for j in 0..M {
                    o[offset + j] = value[j].clone();
                }
            }

            out.into()
        }
    }
    */
}

impl<T> Drop for TensorData<T> {
    fn drop(&mut self) {
        unsafe {
            let len = self.len;
            let slice = slice::from_raw_parts_mut(self.data.as_ptr(), len); 
            drop(Box::from_raw(slice));
        }
    }
}

// unsafe: TensorData is read-only
unsafe impl<T: Send> Send for TensorData<T> {}
unsafe impl<T: Sync> Sync for TensorData<T> {}

///
/// TensorUninit is used to create new tensor data when the caller can guarantee
/// that all items will be initialized.
/// 
/// It is intrinsically unsafe because it's working with uninitialized data.
/// 
pub struct TensorUninit<T=f32> {
    len: usize,

    data: NonNull<T>,
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
    pub unsafe fn as_slice(&self) -> &[T] {
        ptr::slice_from_raw_parts(self.as_ptr(), self.len())
            .as_ref()
            .unwrap()
    }

    #[inline]
    pub unsafe fn as_mut_slice(&mut self) -> &mut [T] {
        ptr::slice_from_raw_parts_mut(self.as_mut_ptr(), self.len())
            .as_mut()
            .unwrap()
    }

    #[inline]
    pub unsafe fn as_sub_slice(&mut self, offset: usize, len: usize) -> &mut [T] {
        assert!(offset <= self.len);
        assert!(offset + len <= self.len);

        ptr::slice_from_raw_parts_mut(self.as_mut_ptr().add(offset), len)
            .as_mut()
            .unwrap()
    }
}

impl<T: Clone + 'static> TensorUninit<T> {
    pub unsafe fn new(len: usize) -> Self {
        let layout = Layout::array::<T>(len).unwrap();
        
        Self {
            len,

            data: NonNull::<T>::new_unchecked(alloc::alloc(layout).cast::<T>()),
        }
    }

    #[inline]
    pub(crate) unsafe fn into(self) -> TensorData<T> {
        let ptr = mem::ManuallyDrop::new(self);
        TensorData::new(ptr.data, ptr.len)
    }

    #[inline]
    pub unsafe fn into_tensor(self, shape: impl Into<Shape>) -> Tensor<T> {
        Tensor::from_uninit(self, shape)
    }

    pub unsafe fn into_tensor_with_id(
        self, 
        shape: impl Into<Shape>, 
        id: TensorId
    ) -> Tensor<T> {
        Tensor::from_uninit_with_id(self, shape, id)
    }

    pub unsafe fn init<F>(&mut self, mut f: F) 
    where
        F: FnMut() -> T
    {
        let len = self.len;

        let slice = self.as_mut_slice();

        for i in 0..len {
            slice[i] = (f)();
        }
    }
}

impl<T> Drop for TensorUninit<T> {
    fn drop(&mut self) {
        unsafe {
            let layout = Layout::array::<T>(self.len()).unwrap();
            alloc::dealloc(self.data.as_ptr().cast::<u8>(), layout);
        }
    }
}

#[cfg(test)]
mod test {
    use std::sync::{Arc, Mutex};

    use crate::{prelude::*, tensor::{Dtype, TensorUninit, data::TensorData}};

    #[test]
    fn data_drop_from_vec() {
        let (p1, p2) = {
            let t1 = Test::new(1);
            let p1 = t1.ptr.clone();
            
            let t2 = Test::new(2);
            let p2 = t2.ptr.clone();
            
            let vec = vec![t1, t2];

            let _data = TensorData::<Test>::from_vec(vec);

            (p1, p2)
        };

        assert_eq!(take(&p1), "Drop[1]");
        assert_eq!(take(&p2), "Drop[2]");
    }

    #[test]
    fn data_drop_from_row_vec() {
        let (p1, p2) = {
            let t1 = Test::new(1);
            let p1 = t1.ptr.clone();
            
            let t2 = Test::new(2);
            let p2 = t2.ptr.clone();
            
            let vec = vec![[t1, t2]];

            let _data = TensorData::<Test>::from_vec_rows(vec);

            (p1, p2)
        };

        assert_eq!(take(&p1), "Drop[1]");
        assert_eq!(take(&p2), "Drop[2]");
    }

    #[test]
    fn data_drop_from_boxed_slice() {
        let (p1, p2) = {
            let t1 = Test::new(1);
            let p1 = t1.ptr.clone();
            
            let t2 = Test::new(2);
            let p2 = t2.ptr.clone();
            
            let slice = Box::new([t1, t2]);

            let _data = TensorData::<Test>::from_boxed_slice(slice);

            (p1, p2)
        };

        assert_eq!(take(&p1), "Drop[1]");
        assert_eq!(take(&p2), "Drop[2]");
    }

    #[test]
    fn data_drop_from_slice() {
        let (p1, p2) = {
            let t1 = Test::new(1);
            let p1 = t1.ptr.clone();
            
            let t2 = Test::new(2);
            let p2 = t2.ptr.clone();
            
            let slice = [t1, t2];

            let _data = TensorData::<Test>::from_slice(&slice);

            (p1, p2)
        };

        assert_eq!(take(&p1), "Drop[1], Drop[1]");
        assert_eq!(take(&p2), "Drop[2], Drop[2]");
    }

    #[test]
    fn drop_tensor() {
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

    #[test]
    fn test_vec_align()
    {
        let mut vec = Vec::<(u8, u32)>::new();

        vec.push((0x01, 0x4000_1000));
        vec.push((0x02, 0x5000_1001));
        vec.push((0x03, 0x6000_1002));
        vec.push((0x04, 0x7000_1003));

        unsafe {
            let tensor = TensorData::from_vec(vec);

            assert_eq!(*tensor.as_ptr().add(0), (0x01, 0x4000_1000));
            assert_eq!(*tensor.as_ptr().add(1), (0x02, 0x5000_1001));
            assert_eq!(*tensor.as_ptr().add(2), (0x03, 0x6000_1002));
            assert_eq!(*tensor.as_ptr().add(3), (0x04, 0x7000_1003));

            assert_eq!(tensor.as_sub_slice(0, tensor.len()), &[
                (0x01, 0x4000_1000),
                (0x02, 0x5000_1001),
                (0x03, 0x6000_1002),
                (0x04, 0x7000_1003)
            ]);
        }

        let mut vec = Vec::<(u32, u8)>::new();

        vec.push((0x4000_1000, 0x01));
        vec.push((0x5000_1001, 0x02));
        vec.push((0x6000_1002, 0x03));
        vec.push((0x7000_1003, 0x04));

        unsafe {
            let tensor = TensorData::from_vec(vec);
            assert_eq!(*tensor.as_ptr().add(0), (0x4000_1000, 0x01));
            assert_eq!(*tensor.as_ptr().add(1), (0x5000_1001, 0x02));
            assert_eq!(*tensor.as_ptr().add(2), (0x6000_1002, 0x03));
            assert_eq!(*tensor.as_ptr().add(3), (0x7000_1003, 0x04));

            assert_eq!(tensor.as_sub_slice(0, tensor.len()), &[
                (0x4000_1000, 0x01),
                (0x5000_1001, 0x02),
                (0x6000_1002, 0x03),
                (0x7000_1003, 0x04)
            ]);
        }
    }

    #[test]
    fn test_vec_rows_basic()
    {
        let mut vec = Vec::<[u32; 3]>::new();

        vec.push([0x01, 0x1000, 0x01_0000]);
        vec.push([0x02, 0x2000, 0x02_0000]);
        vec.push([0x03, 0x3000, 0x03_0000]);
        vec.push([0x04, 0x4000, 0x04_0000]);

        unsafe {
            let data = TensorData::from_vec_rows(vec);
            assert_eq!(data.as_sub_slice(0, data.len()), &[
                0x01, 0x1000, 0x01_0000,
                0x02, 0x2000, 0x02_0000,
                0x03, 0x3000, 0x03_0000,
                0x04, 0x4000, 0x04_0000,
            ]);
        }
    }

    #[test]
    fn test_vec_matrices_basic()
    {
        let mut vec = Vec::<[[u32; 2]; 3]>::new();

        vec.push([[10, 20], [110, 120], [210, 220]]);
        vec.push([[11, 21], [111, 121], [211, 221]]);
        vec.push([[12, 22], [112, 122], [212, 222]]);

        unsafe {
            let data = TensorData::from_vec_matrices(vec);
            assert_eq!(data.as_sub_slice(0, data.len()), &[
                10, 20, 110, 120, 210, 220,
                11, 21, 111, 121, 211, 221,
                12, 22, 112, 122, 212, 222,
            ]);
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

    #[test]
    fn data_as_slice() {
        let data = TensorData::from_vec(vec![1., 2., 3.]);
        unsafe { assert_eq!(data.as_sub_slice(0, 3), &[1., 2., 3.]); }

        let data = TensorData::from_vec(vec![1., 2., 3.]);
        unsafe { assert_eq!(data.as_sub_slice(0, 2), &[1., 2.]); }

        let data = TensorData::from_vec(vec![1., 2., 3.]);
        unsafe { assert_eq!(data.as_sub_slice(1, 2), &[2., 3.]); }

        let data = TensorData::from_vec(vec![1., 2., 3.]);
        unsafe { assert_eq!(data.as_sub_slice(1, 0), &[]); }
    }

    #[test]
    fn uninit_slice_mut() {
        unsafe {
            let mut a = TensorUninit::<f32>::new(1);
            let slice = a.as_mut_slice();
            slice[0] = 10.0;
            assert_eq!(slice, &[10.]);
            let data = a.into();

            assert_eq!(data.as_sub_slice(0, data.len()), &[10.]);
        };
    }

    #[test]
    fn uninit_drop() {
        unsafe {
            let mut uninit = TensorUninit::<u32>::new(1);
            uninit.as_mut_slice()[0] = 3;
            //uninit.init()
        };
    }

    fn take(ptr: &Arc<Mutex<Vec<String>>>) -> String {
        let vec : Vec<String> = ptr.lock().unwrap().drain(..).collect();

        vec.join(", ")
    }

    #[derive(Debug, Clone)]
    struct Test {
        id: usize,
        ptr: Arc<Mutex<Vec<String>>>,
    }

    impl Test {
        fn new(id: usize) -> Self {
            Self {
                id,
                ptr: Arc::new(Mutex::new(Vec::default())),
            }
        }
    }

    impl Dtype for Test {}

    impl Drop for Test {
        fn drop(&mut self) {
            self.ptr.lock().unwrap().push(format!("Drop[{:?}]", self.id));
        }
    }
}