use core::fmt;
use std::{any::type_name, sync::Arc, ops::{Deref}, slice};

use super::{data::TensorData, TensorUninit, slice::TensorSlice, Shape};

pub trait Dtype : Clone + Send + Sync + fmt::Debug + 'static {}

pub struct Tensor<T=f32> {
    id: TensorId,
    shape: Shape,
    offset: usize,
    len: usize,

    data: Arc<TensorData<T>>,
}
/*
impl<T: 'static> Tensor<T> {
    pub(crate) unsafe fn from_data(
        data: TensorData<T>, 
        shape: impl Into<Shape>, 
        id: TensorId,
    ) -> Self {
        let shape = shape.into();
        let len: usize = shape.size();
        
        assert_eq!(len, data.len());
        // data.checkcast::<T>(len);

        Self {
            id,

            shape,
            offset: 0,
            len,

            data: Arc::new(data),
        }
    }
}
*/

impl<T: Clone + 'static> Tensor<T> {
    pub fn empty() -> Self {
        Self {
            id: TensorId::NONE,

            shape: Shape::from([0]),
            offset: 0,
            len: 0,

            data: Arc::new(TensorData::from_slice(&[])),
        }
    }

    pub fn from_slice(data: &[T]) -> Self {
        assert!(data.len() > 0);

        Self {
            id: TensorId::NONE,

            shape: Shape::from(data.len()),
            offset: 0,
            len: data.len(),

            data: Arc::new(TensorData::from_slice(data)),
        }
    }

    pub fn from_vec(vec: Vec<T>, shape: impl Into<Shape>) -> Self {
        //assert!(vec.len() > 0);

        Self {
            id: TensorId::NONE,

            shape: shape.into(),
            offset: 0,
            len: vec.len(),

            data: Arc::new(TensorData::from_vec(vec)),
        }
    }

    pub fn append(&self, tensor: impl Into<Tensor<T>>) -> Tensor<T> {
        let tensor = tensor.into();

        assert_eq!(self.shape().sublen(1..), tensor.shape().sublen(1..));

        let mut vec = Vec::from(self.shape.as_slice());
        vec[0] = self.dim(0) + tensor.dim(0);

        Tensor::from_merge(vec![self.clone(), tensor], vec, TensorId::NONE)
    }
}

impl<T: Clone + 'static> Tensor<T> {
    pub fn from_uninit(data: TensorUninit<T>, shape: impl Into<Shape>) -> Self {
        Self::from_uninit_with_id(data, shape, TensorId::NONE)
    }

    pub fn from_uninit_with_id(
        data: TensorUninit<T>, 
        shape: impl Into<Shape>,
        id: TensorId,
    ) -> Self {
        let shape = shape.into();
        let len: usize = data.len(); // max(1, shape.size());

        assert_eq!(
            data.len(), len, 
            "Tensor data len={} must match shape size {:?}", data.len(), shape.as_slice()
        );

        Self {
            id,

            shape,
            offset: 0,
            len,

            data: Arc::new(data.init()),
        }
    }

    pub fn from_merge(
        vec: Vec<Tensor<T>>, 
        shape: impl Into<Shape>,
        id: TensorId,
    ) -> Self {
        let shape = shape.into();

        let len: usize = vec.iter().map(|t| t.len()).sum();

        assert_eq!(
            len, shape.size(),
            "Tensor data len={} must match shape size {:?}", len, shape.as_slice()
        );

        unsafe {
            let mut uninit = TensorUninit::<T>::new(len);

            let mut o_ptr = uninit.as_mut_ptr();

            for tensor in vec {
                let x = tensor.as_slice();

                for i in 0..tensor.len() {
                    *o_ptr.add(i) = x[i].clone();
                }

                o_ptr = o_ptr.add(tensor.len());
            }

            Self {
                id,

                shape,
                offset: 0,
                len,

                data: Arc::new(uninit.init()),
            }
        }
    }
}

impl<T> Tensor<T> {
    pub(crate) fn with_id(self, id: TensorId) -> Tensor<T> {
        Self { id, ..self }
    }

    pub(crate) fn clone_with_shape(&self, shape: impl Into<Shape>, id: TensorId) -> Tensor<T> {
        let shape = shape.into();
        assert_eq!(shape.size(), self.len());

        Self { 
            id, 
            shape, 
            data: self.data.clone(),
            offset: self.offset,
            len: self.len,
        }
    }

    #[inline]
    pub fn id(&self) -> TensorId {
        self.id
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    #[inline]
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    #[inline]
    pub fn dim(&self, i: usize) -> usize {
        self.shape.dim(i)
    }

    #[inline]
    pub fn dim_tail(&self) -> usize {
        self.shape.dim_tail()
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.shape.cols()
    }

    #[inline]
    pub fn rows(&self) -> usize {
        self.shape.rows()
    }

    #[inline]
    pub fn batch_len(&self, base_rank: usize) -> usize {
        self.shape.batch_len(base_rank)
    }

    #[inline]
    pub fn broadcast(&self, b: &Self) -> usize {
        self.shape.broadcast(b.shape())
    }

    #[inline]
    pub fn broadcast_min(
        &self, 
        a_min: usize, 
        b: &Self, 
        b_min: usize
    ) -> usize {
        self.shape.broadcast_min(a_min, b.shape(), b_min)
    }

    pub fn reshape(&self, shape: impl Into<Shape>) -> Tensor<T> {
        let shape = shape.into();

        assert_eq!(self.len(), shape.size());

        // TODO: reshape should probably have Op

        Tensor {
            id: self.id,

            shape,
            offset: self.offset,
            len: self.len,

            data: self.data.clone(),
        }
    }
    /*
    #[inline]
    pub fn data(&self) -> &Arc<TensorData> {
        &self.data
    }
    */

    #[inline]
    pub fn get(&self, offset: usize) -> Option<&T> {
        unsafe { self.data.get(self.offset + offset) }
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { self.data.as_sub_slice(self.offset, self.len) }
    }

    // Returns a possibly-wrapped pointer at the offset to support
    // broadcast
    #[inline]
    pub unsafe fn as_wrap_slice(&self, offset: usize) -> &[T] {
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
    pub unsafe fn as_wrap_ptr(&self, offset: usize) -> *const T {
        if offset < self.len {
            self.data.as_ptr().add(self.offset + offset)
        } else {
            self.data.as_ptr().add(self.offset + offset % self.len)
        }
    }

    pub fn subslice_flat(&self, offset: usize, len: usize, shape: impl Into<Shape>) -> Self {
        assert!(offset <= self.len);
        assert!(offset + len <= self.len);

        let shape = shape.into();

        let shape_len : usize = shape.size();
        assert!(shape_len == len || shape.size() == 0 && len == 1);

        Self {
            id: TensorId::NONE,

            shape,

            offset: self.offset + offset,
            len,

            data: self.data.clone(),
        }
    }

    // TODO: reparam to use range
    pub fn subslice(&self, offset: usize, len: usize) -> Self {
        let dim_0 = self.dim(0);

        assert!(offset <= dim_0);
        assert!(offset + len <= dim_0);

        let size : usize = self.shape().as_slice()[1..].iter().product();

        let mut shape = Vec::from(self.shape.as_slice());
        shape[0] = len;

        self.subslice_flat(offset * size, len * size, shape)
    }

    #[inline]
    pub fn iter(&self) -> slice::Iter<T> {
        self.as_slice().iter()
    }

    #[inline]
    pub fn iter_slice(&self) -> slice::ChunksExact<T> {
        let dim = self.dim_tail();

        if self.rank() > 1 {
            self.as_slice().chunks_exact(dim).into_iter()
        } else {
            self.as_slice().chunks_exact(1).into_iter()
        }
    }
}

impl<T:Clone> Tensor<T> {
    pub fn slice<S:TensorSlice>(&self, index: S) -> Tensor<T> {
        S::slice(index, &self)
    }
}

impl<T> Deref for Tensor<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        Self { 
            id: self.id,

            shape: self.shape.clone(), 

            offset: self.offset,
            len: self.len,

            data: self.data.clone(),
        }
    }
}

impl<T:PartialEq> PartialEq for Tensor<T> {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.as_slice() == other.as_slice()
    }
}

impl<T: fmt::Debug> fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor<{}> {{", type_name::<T>())?;

        if self.shape.rank() > 1 {
            write!(f, "\n")?;
        }

        fmt_tensor_rec(&self, f, self.rank(), 0)?;
        
        write!(f, ", shape: {:?}", &self.shape.as_slice())?;
        // write!(f, ", dtype: {}", type_name::<T>())?;

        if f.alternate() && self.id().is_some() {
            write!(f, ", id: {:#?}", &self.id)?;
        }

        write!(f, "}}")?;
        Ok(())
    }
}

fn fmt_tensor_rec<T:fmt::Debug>(
    tensor: &Tensor<T>, 
    f: &mut fmt::Formatter<'_>, 
    rank: usize,
    offset: usize
) -> fmt::Result {
    match rank {
        0 => write!(f, "{:?}", tensor[offset]),
        1 => {
            write!(f, "[")?;

            for j in 0..tensor.cols() {
                if j > 0 {
                    write!(f, " ")?;
                }

                fmt_tensor_rec(tensor, f, rank - 1, offset + j)?;
            }

            write!(f, "]")
        },
        2 => {
            write!(f, "[")?;

            let shape = tensor.shape();

            let stride = shape.cols();
            for j in 0..shape.rows() {
                if j > 0 {
                    write!(f, ",\n ")?;
                }

                fmt_tensor_rec::<T>(tensor, f, rank - 1, offset + j * stride)?;
            }

            write!(f, "]")
        },
        n => {
            write!(f, "[")?;

            let shape = tensor.shape();
            let rank = shape.rank();
            // TODO:
            let stride : usize = shape.sublen((rank - n + 1)..);
            for j in 0..shape.dim_rev(n - 1) {
                if j > 0 {
                    write!(f, ",\n\n  ")?;
                }

                fmt_tensor_rec::<T>(tensor, f, n - 1, offset + j * stride)?;
            }

            write!(f, "]")
        },
    }
}

impl<T: Dtype> From<&Tensor<T>> for Tensor<T> {
    fn from(value: &Tensor<T>) -> Self {
        value.clone()
    }
}

impl<T:Dtype> From<T> for Tensor<T> {
    fn from(value: T) -> Self {
        Tensor::from_vec(vec![value], Shape::scalar())
    }
}

impl<T: Dtype> From<()> for Tensor<T> {
    // TODO: possible conflict with Tensors
    fn from(_value: ()) -> Self {
        Tensor::empty()
    }
}

impl<T:Dtype> From<Vec<T>> for Tensor<T> {
    fn from(value: Vec<T>) -> Self {
        let len = value.len();

        Tensor::<T>::from_vec(value, Shape::from(len))
    }
}

impl<T:Dtype, const N:usize> From<[T; N]> for Tensor<T> {
    fn from(value: [T; N]) -> Self {
        // TODO: avoid copy
        let vec = Vec::<T>::from(value);
        let len = vec.len();

        Tensor::from_vec(vec, Shape::from(len))
    }
}

impl<T: Dtype> From<&[T]> for Tensor<T> {
    fn from(value: &[T]) -> Self {
        // TODO: avoid copy
        let vec = Vec::<T>::from(value);
        let len = vec.len();

        Tensor::from_vec(vec, Shape::from(len))
    }
}

impl From<&str> for Tensor<String> {
    fn from(value: &str) -> Self {
        Tensor::from_vec(vec![value.to_string()], Shape::scalar())
    }
}

impl From<Vec<&str>> for Tensor<String> {
    fn from(value: Vec<&str>) -> Self {
        let len = value.len();

        let vec = value.iter().map(|s| s.to_string()).collect();

        Tensor::<String>::from_vec(vec, Shape::from(len))
    }
}

impl<const N:usize> From<[&str; N]> for Tensor<String> {
    fn from(value: [&str; N]) -> Self {
        let vec : Vec<String> = value.iter().map(|s| s.to_string()).collect();
        let len = vec.len();

        Tensor::from_vec(vec, Shape::from(len))
    }
}

impl<T, const N: usize, const M: usize> From<[[T; N]; M]> for Tensor<T>
where
    T: Dtype,
{
    fn from(value: [[T; N]; M]) -> Self {
        let mut vec = Vec::<T>::new();

        for value in value.iter() {
            for value in value.iter() {
                vec.push(value.clone());
            }
        }

        Tensor::from_vec(vec, [M, N])
    }
}

impl<T, const N: usize, const M: usize, const L: usize>
    From<[[[T; N]; M]; L]> for Tensor<T>
where
    T: Dtype
{
    fn from(value: [[[T; N]; M]; L]) -> Self {
        let mut vec = Vec::<T>::new();

        for value in value.iter() {
            for value in value.iter() {
                for value in value.iter() {
                    vec.push(value.clone());
                }
            }
        }

        Tensor::from_vec(vec, [L, M, N])
    }
}

impl<T, const N: usize, const M: usize, const L: usize, const K: usize>
    From<[[[[T; N]; M]; L]; K]> for Tensor<T>
where
    T:Dtype
{
    fn from(value: [[[[T; N]; M]; L]; K]) -> Self {
        let mut vec = Vec::<T>::new();

        for value in value {
            for value in value {
                for value in value {
                    for value in value {
                        vec.push(value.clone())
                    }
                }
            }
        }

        Tensor::from_vec(vec, [K, L, M, N])
    }
}

impl<T:Dtype> FromIterator<T> for Tensor<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let vec : Vec<T> = Vec::from_iter(iter);
        let len = vec.len();

        Tensor::from_vec(vec, len)
    }
}

impl<T:Dtype + Copy + 'static> From<&Vec<Tensor<T>>> for Tensor<T> {
    fn from(values: &Vec<Tensor<T>>) -> Self {
        Tensor::<T>::from(values.as_slice())
    }
}

impl<T:Dtype + Copy + 'static> From<&[Tensor<T>]> for Tensor<T> {
    fn from(values: &[Tensor<T>]) -> Self {
        let n = values.len();
        assert!(n > 0);

        let sublen = values[0].len();

        let shape = values[0].shape();

        for value in values {
            assert_eq!(sublen, value.len(), "tensor length must match");
            assert_eq!(shape, value.shape(), "tensor shapes must match");
        }

        unsafe {
            let mut data = TensorUninit::<T>::new(sublen * n);

            for j in 0..n {
                let tensor = &values[j];

                for (i, value) in tensor.as_slice().iter().enumerate() {
                    data[j * sublen + i] = value.clone();
                }
            }

            Tensor::from_uninit(data, shape.push(n))
        }
    }
}

impl<T:Dtype + Copy + 'static, const N: usize> From<[Tensor<T>; N]> for Tensor<T> {
    fn from(values: [Tensor<T>; N]) -> Self {
        let vec = Vec::from(values);

        Tensor::from(&vec)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(u32, u32);

impl TensorId {
    pub const NONE : TensorId = TensorId(u32::MAX, u32::MAX);

    #[inline]
    pub(crate) fn new(model_index: u32, tensor_index: u32) -> TensorId {
        TensorId(model_index, tensor_index)
    }

    #[inline]
    pub fn index(&self) -> usize {
        self.1 as usize
    }

    #[inline]
    pub fn model_index(&self) -> usize {
        self.0 as usize
    }

    #[inline]
    pub fn is_some(&self) -> bool {
        self != &Self::NONE
    }

    #[inline]
    pub fn is_none(&self) -> bool {
        self == &Self::NONE
    }

    pub fn unset() -> TensorId {
        Self::NONE
    }
}

impl fmt::Debug for TensorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_none() {
            write!(f, "TensorId(None)")
        } else {
            write!(f, "TensorId({}:{})", self.0, self.1)
        }
    }
}

//trait Dtype : Copy {}
impl Dtype for bool {}
impl Dtype for u8 {}
impl Dtype for i32 {}
impl Dtype for u32 {}
impl Dtype for usize {}
impl Dtype for f32 {}
impl Dtype for String {}

#[cfg(test)]
mod test {
    use tensor::Shape;

    use crate::{tensor};

    use super::{Tensor};

    #[test]
    fn debug_tensor_from_f32() {
        let t = Tensor::from(10.5);
        assert_eq!(format!("{:?}", t), "Tensor {10.5, shape: [], dtype: f32");
    }

    #[test]
    fn debug_vector_from_slice_f32() {
        //let t = Tensor::from([]);
        //assert_eq!(format!("{:?}", t), "Tensor{[], shape: [0], dtype: f32}");

        let t = Tensor::from([10.5]);
        assert_eq!(format!("{:?}", t), "Tensor{[10.5], shape: [1], dtype: f32}");

        let t = Tensor::from([1., 2.]);
        assert_eq!(format!("{:?}", t), "Tensor{[1 2], shape: [2], dtype: f32}");

        let t = Tensor::from([1., 2., 3., 4., 5.]);
        assert_eq!(format!("{:?}", t), "Tensor{[1 2 3 4 5], shape: [5], dtype: f32}");
    }

    #[test]
    fn debug_matrix_from_slice_f32() {
        //let t = Tensor::from([[]]);
        //assert_eq!(format!("{:?}", t), "Tensor{\n[[]], shape: [1, 0], dtype: f32}");

        let t = Tensor::from([[10.5]]);
        assert_eq!(format!("{:?}", t), "Tensor{\n[[10.5]], shape: [1, 1], dtype: f32}");

        let t = Tensor::from([[1., 2.]]);
        assert_eq!(format!("{:?}", t), "Tensor{\n[[1 2]], shape: [1, 2], dtype: f32}");

        let t = Tensor::from([[1., 2., 3.], [4., 5., 6.]]);
        assert_eq!(format!("{:?}", t), "Tensor{\n[[1 2 3],\n [4 5 6]], shape: [2, 3], dtype: f32}");
    }

    #[test]
    fn debug_tensor3_from_slice_f32() {
        //let t = Tensor::from([
        //    [[]]
        //]);
        //assert_eq!(format!("{:?}", t), "Tensor{\n[[[]]], shape: [1, 1, 0], dtype: f32}");

        let t = Tensor::<f32>::from([
            [[10.5]]
        ]);
        assert_eq!(format!("{:?}", t), "Tensor{\n[[[10.5]]], shape: [1, 1, 1], dtype: f32}");

        let t = Tensor::<f32>::from([
            [[1., 2.]],
            [[101., 102.]]
        ]);
        assert_eq!(format!("{:?}", t), "Tensor{\n[[[1 2]],\n\n  [[101 102]]], shape: [2, 1, 2], dtype: f32}");

        let t = Tensor::<f32>::from([
            [[1.0, 2.], [3., 4.]],
            [[101., 102.], [103., 104.]]
        ]);
        assert_eq!(format!("{:?}", t), "Tensor{\n[[[1 2],\n [3 4]],\n\n  [[101 102],\n [103 104]]], shape: [2, 2, 2], dtype: f32}");
    }

    #[test]
    fn debug_vector_from_macro() {
        let t = tensor!(1.);
        assert_eq!(format!("{:?}", t), "Tensor {1.0, shape: [], dtype: f32}");

        let t = tensor!([1.]);
        assert_eq!(format!("{:?}", t), "Tensor {[1.0], shape: [1], dtype: f32}");

        let t = tensor!([1., 2.]);
        assert_eq!(format!("{:?}", t), "Tensor {[1.0 2.0], shape: [2], dtype: f32}");

        let t = tensor!([[1., 2., 3.], [3., 4., 5.]]);
        assert_eq!(format!("{:?}", t), "Tensor {\n[[1.0 2.0 3.0],\n [3.0 4.0 5.0]], shape: [3, 2], dtype: f32}");

        let t = tensor!([
            [[1., 2.], [3., 4.]],
            [[11., 12.], [13., 14.]]
        ]);
        assert_eq!(format!("{:?}", t), "Tensor {\n[[[1.0 2.0],\n [3.0 4.0]],\n\n  [[11.0 12.0],\n [13.0 14.0]]], shape: [2, 2, 2], dtype: f32}");
    }

    #[test]
    fn tensor_0_from_scalar() {
        let t0 : Tensor = 0.25.into();

        assert_eq!(t0.len(), 1);
        assert_eq!(t0.shape(), &Shape::scalar());
        assert_eq!(t0.get(0), Some(&0.25));
    }

    #[test]
    fn tensor_from_f32() {
        let t = Tensor::from(10.5);
        assert_eq!(t.len(), 1);
        assert_eq!(t[0], 10.5);
        assert_eq!(t.as_slice(), &[10.5]);
        assert_eq!(t.shape(), &Shape::scalar());
    }

    #[test]
    fn tensor_from_scalar_iterator() {
        let t0 = Tensor::from_iter(0..6);
        assert_eq!(t0.len(), 6);
        assert_eq!(t0.shape().as_slice(), &[6]);
        for i in 0..6 {
            assert_eq!(t0.get(i), Some(&i));
        }
    }

    #[test]
    fn tensor_from_tensor_slice() {
        let t0 = Tensor::from([tensor!(2.), tensor!(1.), tensor!(3.)]);
        assert_eq!(t0.len(), 3);
        assert_eq!(t0.shape().as_slice(), &[3]);
        assert_eq!(t0.get(0), Some(&2.));
        assert_eq!(t0.get(1), Some(&1.));
        assert_eq!(t0.get(2), Some(&3.));

        let t1 = Tensor::from([
            tensor!([1., 2.]), 
            tensor!([2., 3.]), 
            tensor!([3., 4.])]
        );
        assert_eq!(t1.len(), 6);
        assert_eq!(t1.shape().as_slice(), &[2, 3]);
        assert_eq!(t1[0], 1.);
        assert_eq!(t1[1], 2.);
        assert_eq!(t1[2], 2.);
        assert_eq!(t1[3], 3.);
        assert_eq!(t1[4], 3.);
        assert_eq!(t1[5], 4.);
    }

    #[test]
    fn tensor_from_vec_slice() {
        let vec = vec![tensor!(2.), tensor!(1.), tensor!(3.)];

        let t0 = Tensor::from(vec.as_slice());
        assert_eq!(t0.len(), 3);
        assert_eq!(t0.shape().as_slice(), &[3]);
        assert_eq!(t0.get(0), Some(&2.));
        assert_eq!(t0.get(1), Some(&1.));
        assert_eq!(t0.get(2), Some(&3.));

        let vec = vec![
            tensor!([1., 2.]), 
            tensor!([2., 3.]), 
            tensor!([3., 4.])
        ];

        let ptr = vec.as_slice();
        let t1 = Tensor::from(ptr);

        assert_eq!(t1.len(), 6);
        assert_eq!(t1.shape().as_slice(), &[2, 3]);
        assert_eq!(t1[0], 1.);
        assert_eq!(t1[1], 2.);
        assert_eq!(t1[2], 2.);
        assert_eq!(t1[3], 3.);
        assert_eq!(t1[4], 3.);
        assert_eq!(t1[5], 4.);
        
        let t1 = Tensor::from(&vec);

        assert_eq!(t1.len(), 6);
        assert_eq!(t1.shape().as_slice(), &[2, 3]);
        assert_eq!(t1[0], 1.);
        assert_eq!(t1[1], 2.);
        assert_eq!(t1[2], 2.);
        assert_eq!(t1[3], 3.);
        assert_eq!(t1[4], 3.);
        assert_eq!(t1[5], 4.);
    }

    #[test]
    fn shape_from_zeros() {
        let t = Tensor::zeros([3, 2, 4, 5]);
        assert_eq!(t.shape().as_slice(), &[3, 2, 4, 5]);
        assert_eq!(t.rank(), 4);
        assert_eq!(t.cols(), 5);
        assert_eq!(t.rows(), 4);
        assert_eq!(t.batch_len(2), 6);
        assert_eq!(t.len(), 3 * 2 * 4 * 5);
    }

    #[test]
    fn string_tensor_from_macro() {
        let t = tensor!("test");
        assert_eq!(t.shape().as_slice(), &[]);

        assert_eq!(&t[0], "test");

        let t = tensor!(["t1", "t2", "t3"]);
        assert_eq!(t.shape().as_slice(), &[3]);

        assert_eq!(&t[0], "t1");
        assert_eq!(&t[1], "t2");
        assert_eq!(&t[2], "t3");
    }

    #[test]
    fn tensor_iter() {
        let vec : Vec<u32> = tensor!([1, 2, 3, 4]).iter().map(|v| *v).collect();
        let vec2 : Vec<u32> = vec!(1, 2, 3, 4);
        assert_eq!(vec, vec2);

        let vec : Vec<u32> = tensor!([[1, 2], [3, 4]]).iter().map(|v| *v).collect();
        let vec2 : Vec<u32> = vec!(1, 2, 3, 4);
        assert!(vec.iter().zip(vec2.iter()).all(|(x, y)| x == y));
    }

    #[test]
    fn tensor_iter_slice() {
        let vec : Vec<Vec<u32>> = tensor!([1, 2, 3, 4]).iter_slice().map(|v| Vec::from(v)).collect();
        let vec2 : Vec<Vec<u32>> = vec!(vec!(1), vec!(2), vec!(3), vec!(4));
        assert_eq!(vec, vec2);

        let vec : Vec<Vec<u32>> = tensor!([[1, 2], [3, 4]]).iter_slice().map(|v| Vec::from(v)).collect();
        let vec2 : Vec<Vec<u32>> = vec!(vec![1, 2], vec![3, 4]);
        assert_eq!(vec, vec2);
    }
}