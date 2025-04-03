use core::slice;
use std::{
    alloc::{self, Layout}, 
    mem, 
    ptr::{self, NonNull},
};

use crate::Tensor;

use super::Shape;

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

    pub(crate) fn into_tensor(self, shape: impl Into<Shape>) -> Tensor<T> {
        Tensor::from_data(self, shape)
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

impl<T: 'static> TensorData<T> {
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

impl<T: 'static> TensorData<T> {
    /// This initialization is unsafe because the data is uninitialized,
    /// which means the caller must take take to both assign every item,
    /// and to only use o.write(data) to update values because other methods 
    /// will try to drop the uninitialized previous
    pub(crate) unsafe fn unsafe_init<F>(len: usize, mut init: F) -> TensorData<T>
    where
        F: FnMut(*mut T)
    {
        let mut uninit = TensorUninit::new(len);

        (init)(uninit.as_mut_ptr());

        uninit.into()
    }

    pub(super) fn map<U>(
        a: &Tensor<U>,
        mut f: impl FnMut(&U) -> T
    ) -> Self {
        let len = a.len();
        
        unsafe {
            Self::unsafe_init(len, |o| {
                let a = a.as_slice();
            
                for i in 0..len {
                    o.add(i).write((f)(&a[i]));
                }
            })
        }
    }

    pub(crate) fn map2<U, V, F>(
        a: &Tensor<U>,
        b: &Tensor<V>,
        mut f: F
    ) -> TensorData<T>
    where
        F: FnMut(&U, &V) -> T
    {
        let a_len = a.len();
        let b_len = b.len();

        let size = a_len.max(b_len);
        let inner = a_len.min(b_len);
        let batch = size / inner;

        assert!(batch * inner == size, "broadcast mismatch a.len={} b.len={}", a_len, b_len);
        
        unsafe {
            Self::unsafe_init(size, |o| {
                for n in 0..batch {
                    let offset = n * inner;

                    let a = a.as_wrap_slice(offset);
                    let b = b.as_wrap_slice(offset);

                    for k in 0..inner {
                        o.add(offset + k).write(f(&a[k], &b[k]));
                    }
                }
            })
        }
    }

    pub(crate) fn map_slice<const M: usize, U>(
        a: &Tensor<U>, 
        n: usize,
        f: impl Fn(&[U]) -> [T; M]
    ) -> Self {
        assert!(n > 0);

        let len = a.len() / n;
        assert!(n * len == a.len());

        unsafe {
            Self::unsafe_init(M * len, |o| {
                let a = a.as_ptr();

                for i in 0..len {
                    let offset = n * i;

                    let slice = ptr::slice_from_raw_parts(a.add(offset), n)
                        .as_ref()
                        .unwrap();

                    let value = (f)(slice);

                    for (j, value) in value.into_iter().enumerate() {
                        o.add(offset + j).write(value);
                    }
                }
            })
        }
    }

    pub(super) fn fold_into<U, S, F>(
        tensor: &Tensor<U>,
        cols: usize,
        init: S,
        mut f: F,
    ) -> TensorData<T> 
    where
        S: Clone + Into<T>,
        F: FnMut(S, &U) -> S,
    {
        let len = tensor.len() / cols;
        assert!(cols * len == tensor.len());
        
        unsafe {
            Self::unsafe_init(len, |o| {
                let a = tensor.as_slice();

                for j in 0..len {
                    let mut value = init.clone();

                    let offset = j * cols;

                    for i in 0..cols {
                        value = (f)(value, &a[offset + i]);
                    }

                    o.add(j).write(value.into());
                }
            })
        }
    }
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

impl<T: 'static> TensorUninit<T> {
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

    pub(crate) unsafe fn create<F>(len: usize, mut init: F) -> TensorData<T>
    where
        F: FnMut(&mut [T])
    {
        let mut uninit = Self::new(len);

        (init)(uninit.as_mut_slice());

        uninit.into()
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
        let a = ten!(10.);
        assert_eq!(a.as_slice(), &[10.]);

        let a = ten!([10.]);
        assert_eq!(a.as_slice(), &[10.]);

        let a = ten!([10., 20., 30.]);
        assert_eq!(a.as_slice(), &[10., 20., 30.]);

        let a = ten!([[10., 20.], [30., 40.]]);
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

    #[test]
    fn uninit_deadbeef() {
        // TODO: validate the dropped values, which would require a static global
        unsafe {
            TensorData::<Deadbeef>::unsafe_init(1, |o| {
                o.write(Deadbeef::new(0x10));
                o.write(Deadbeef::new(0x20));
                o.write(Deadbeef::new(0x30));
            });
        }

        unsafe {
            TensorData::<Deadbeef>::unsafe_init(1, |o| {
                o.write(Deadbeef::new(0x10));
                // Note: this drops the previous value, so can't be used
                *o = Deadbeef::new(0x20);
                // Note: this does not drop the previous value
                o.write(Deadbeef::new(0x30));
            });
        }
    }

    #[test]
    fn map_deadbeef() {
        let t = ten!([0x10, 0x20, 0x30]);

        t.map(|v| Deadbeef::new(*v));
    }

    struct Deadbeef {
        data: u32,
    }

    impl Deadbeef {
        fn new(data: u32) -> Self {
            Self {
                data,
            }
        }
    }

    impl Drop for Deadbeef {
        fn drop(&mut self) {
            println!("Drop {:.08x}", self.data);
        }
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