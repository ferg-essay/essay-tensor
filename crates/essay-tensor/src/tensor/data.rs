use core::{slice, fmt};
use std::{ptr::NonNull, alloc::Layout, alloc, ops::{Index, self}};

use super::tensor::Dtype;

pub struct TensorData<D:Dtype=f32> {
    data: NonNull<D>,
    len: usize,
}

pub struct TensorUninit<D:Dtype=f32> {
    data: NonNull<D>,
    len: usize,
}

pub struct Data4<D:Dtype=f32> {
    data: [D; 4]
}

pub struct Data8<D:Dtype=f32> {
    data: [D; 8]
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
    pub unsafe fn get_unchecked(&self, offset: usize) -> D {
        *self.data.as_ptr().add(offset)
    }

    #[inline]
    pub unsafe fn as_ptr(&self) -> *const D {
        self.data.as_ptr()
    }

    #[inline]
    pub fn read_4(&self, offset: usize, inc: usize) -> Data4<D> {
        assert!(offset + 3 < self.len);

        unsafe {
            let ptr = self.data.as_ptr().add(offset);

            Data4 {
                data: [
                    *ptr,
                    *ptr.add(inc),
                    *ptr.add(2 * inc),
                    *ptr.add(3 * inc),
                ]
            }
        }
    }

    #[inline]
    pub fn read_4a(&self, offset: usize, inc: usize) -> [D; 4] {
        assert!(offset + 3 < self.len);

        unsafe {
            let ptr = self.data.as_ptr().add(offset);

            [
                *ptr,
                *ptr.add(inc),
                *ptr.add(2 * inc),
                *ptr.add(3 * inc),
            ]
        }
    }

    #[inline]
    pub fn read_8(&self, offset: usize, inc: usize) -> Data8<D> {
        assert!(offset + 7 < self.len);

        unsafe {
            let ptr = self.data.as_ptr().add(offset);

            Data8 {
                data: [
                    *ptr,
                    *ptr.add(inc),
                    *ptr.add(2 * inc),
                    *ptr.add(3 * inc),

                    *ptr.add(4 * inc),
                    *ptr.add(5 * inc),
                    *ptr.add(6 * inc),
                    *ptr.add(7 * inc),
                ]
            }
        }
    }
}

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

impl<D:Dtype + PartialEq> PartialEq for TensorData<D> {
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }

        for i in 0..self.len {
            if self.get(i) != other.get(i) {
                println!("DIFF {:?} {:?}", self.get(i), other.get(i));
                return false;
            }
        }

        return true;
    }
}

impl<D:Dtype> fmt::Debug for TensorData<D> {
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

    #[inline]
    pub fn read_4(&self, offset: usize) -> Data4<D> {
        assert!(offset + 3 < self.len);

        unsafe {
            let ptr = self.data.as_ptr().add(offset);

            Data4 {
                data: [
                    *ptr,
                    *ptr.add(1),
                    *ptr.add(2),
                    *ptr.add(3),
                ]
            }
        }
    }

    #[inline]
    pub fn read_8(&self, offset: usize) -> Data8<D> {
        assert!(offset + 7 < self.len);

        unsafe {
            let ptr = self.data.as_ptr().add(offset);

            Data8 {
                data: [
                    *ptr,
                    *ptr.add(1),
                    *ptr.add(2),
                    *ptr.add(3),

                    *ptr.add(4),
                    *ptr.add(5),
                    *ptr.add(6),
                    *ptr.add(7),
                ]
            }
        }
    }

    #[inline]
    pub fn write_4(&self, offset: usize, data: &Data4<D>) {
        assert!(offset + 3 < self.len);

        unsafe {
            let ptr = self.data.as_ptr().add(offset);

            *ptr = data.data[0];
            *ptr.add(1) = data.data[1];
            *ptr.add(2) = data.data[2];
            *ptr.add(3) = data.data[3];
        }
    }

    #[inline]
    pub fn write_8(&self, offset: usize, data: &Data8<D>) {
        assert!(offset + 7 < self.len);

        unsafe {
            let ptr = self.data.as_ptr().add(offset);

            *ptr = data.data[0];
            *ptr.add(1) = data.data[1];
            *ptr.add(2) = data.data[2];
            *ptr.add(3) = data.data[3];

            *ptr.add(4) = data.data[4];
            *ptr.add(5) = data.data[5];
            *ptr.add(6) = data.data[6];
            *ptr.add(7) = data.data[7];
        }
    }
}
/*
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
*/

impl<D:Dtype> ops::Deref for TensorUninit<D> {
    type Target = [D];

    #[inline]
    fn deref(&self) -> &[D] {
        unsafe { slice::from_raw_parts(self.as_ptr(), self.len) }
    }
}

impl<D:Dtype> ops::DerefMut for TensorUninit<D> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [D] {
        unsafe { slice::from_raw_parts_mut(self.as_ptr(), self.len) }
    }
}

impl Data4 {
    #[inline]
    pub fn muladd(&self, right: &Data4) -> f32 {
        let v0 = self.data[0] * right.data[0];
        let v1 = self.data[1] * right.data[1];
        let v2 = self.data[2] * right.data[2];
        let v3 = self.data[3] * right.data[3];

        let va = v0 + v2;
        let vb = v1 + v3;

        va + vb
    }
}

impl ops::Add<Data4> for Data4 {
    type Output = Data4;

    #[inline]
    fn add(self, rhs: Data4) -> Self::Output {
        let l_data = &self.data;
        let r_data = &rhs.data;

        Data4 {
            data: [
                l_data[0] + r_data[0],
                l_data[1] + r_data[1],
                l_data[2] + r_data[2],
                l_data[3] + r_data[3],
            ]
        }
    }
}

impl ops::Mul<Data4<f32>> for Data4<f32> {
    type Output = Data4<f32>;

    #[inline]
    fn mul(self, rhs: Data4<f32>) -> Self::Output {
        let l_data = &self.data;
        let r_data = &rhs.data;

        Data4 {
            data: [
                l_data[0] * r_data[0],
                l_data[1] * r_data[1],
                l_data[2] * r_data[2],
                l_data[3] * r_data[3],
            ]
        }
    }
}

impl Data8 {
    #[inline]
    pub fn muladd(&self, right: &Data8) -> f32 {
        let v0 = self.data[0] * right.data[0];
        let v1 = self.data[1] * right.data[1];
        let v2 = self.data[2] * right.data[2];
        let v3 = self.data[3] * right.data[3];

        let v4 = self.data[4] * right.data[4];
        let v5 = self.data[5] * right.data[5];
        let v6 = self.data[6] * right.data[6];
        let v7 = self.data[7] * right.data[7];

        let va = v0 + v4;
        let vb = v1 + v5;
        let vc = v2 + v6;
        let vd = v3 + v7;

        va + vb + vc + vd
    }
}

impl ops::Add<Data8> for Data8 {
    type Output = Data8;

    #[inline]
    fn add(self, rhs: Data8) -> Self::Output {
        let l_data = &self.data;
        let r_data = &rhs.data;

        Data8 {
            data: [
                l_data[0] + r_data[0],
                l_data[1] + r_data[1],
                l_data[2] + r_data[2],
                l_data[3] + r_data[3],

                l_data[4] + r_data[4],
                l_data[5] + r_data[5],
                l_data[6] + r_data[6],
                l_data[7] + r_data[7],
            ]
        }
    }
}

impl ops::Mul<Data8<f32>> for Data8<f32> {
    type Output = Data8<f32>;

    #[inline]
    fn mul(self, rhs: Data8<f32>) -> Self::Output {
        let l_data = &self.data;
        let r_data = &rhs.data;

        Data8 {
            data: [
                l_data[0] * r_data[0],
                l_data[1] * r_data[1],
                l_data[2] * r_data[2],
                l_data[3] * r_data[3],

                l_data[4] * r_data[4],
                l_data[5] * r_data[5],
                l_data[6] * r_data[6],
                l_data[7] * r_data[7],
            ]
        }
    }
}
