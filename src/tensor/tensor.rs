use core::fmt;
use std::{any::type_name, ops::Deref, slice, sync::Arc};

use num_complex::Complex;

use crate::model::{NodeOp, Tape, expr::GradOperation};

use super::{data::{TensorData, TensorDataSlice}, slice::TensorSlice, Shape, TensorUninit};

pub struct Tensor<T=f32> {
    id: TensorId,
    shape: Shape,
    offset: usize,
    len: usize,

    data: Arc<TensorData<T>>,
}

impl<T> Tensor<T> {
    pub(crate) fn from_data(data: TensorData<T>, shape: impl Into<Shape>) -> Self {
        Self {
            id: TensorId::NONE,

            shape: shape.into(),
            offset: 0,
            len: data.len(),

            data: Arc::new(data)
        }
    }
    pub fn from_vec(vec: Vec<T>, shape: impl Into<Shape>) -> Self {
        Self {
            id: TensorId::NONE,

            shape: shape.into(),
            offset: 0,
            len: vec.len(),

            data: Arc::new(TensorData::from_vec(vec)),
        }
    }
}

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

    pub fn append(&self, tensor: impl Into<Tensor<T>>) -> Tensor<T> {
        let tensor = tensor.into();

        assert_eq!(self.shape().sublen(1..), tensor.shape().sublen(1..));

        let mut vec = Vec::from(self.shape.as_slice());
        vec[0] = self.dim(0) + tensor.dim(0);

        Tensor::from_merge(vec![self.clone(), tensor], vec, TensorId::NONE)
    }

    pub unsafe fn from_uninit(data: TensorUninit<T>, shape: impl Into<Shape>) -> Self {
        Self::from_uninit_with_id(data, shape, TensorId::NONE)
    }

    pub unsafe fn from_uninit_with_id(
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

            data: Arc::new(data.into()),
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

                data: Arc::new(uninit.into()),
            }
        }
    }
}

impl<T: Dtype> Tensor<T> {
    pub fn join_vec(
        vec: &Vec<Vec<T>>, 
    ) -> Self {
        let mut flat_vec = Vec::<T>::new();

        for item in vec.iter() {
            for v in item.iter() {
                flat_vec.push(v.clone());
            }
        }

        Tensor::from(flat_vec)
    }
}

impl<T> Tensor<T> {
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
    pub fn cols(&self) -> usize {
        self.shape.dim_tail()
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

    pub(crate) fn with_id(self, id: TensorId) -> Tensor<T> {
        Self { id, ..self }
    }

    pub fn with_shape(self, shape: impl Into<Shape>) -> Tensor<T> {
        let shape = shape.into();

        assert_eq!(shape.size(), self.len(), "shape size must match {:?} new={:?}", 
            self.shape().as_slice(), shape.as_slice()
        );

        Self { shape, ..self }
    }

    pub(crate) fn clone_with_shape(&self, shape: impl Into<Shape>, id: TensorId) -> Tensor<T> {
        let shape = shape.into();
        assert_eq!(shape.size(), self.len(), "shape size must match {:?} new={:?}", 
            self.shape().as_slice(), shape.as_slice()
        );

        Self { 
            id, 
            shape, 
            data: self.data.clone(),
            offset: self.offset,
            len: self.len,
        }
    }

    pub(crate) fn reshape_impl(
        &self, 
        shape: impl Into<Shape>,
        id: TensorId,
    ) -> Self {
        let shape = shape.into();

        assert_eq!(self.len(), shape.size());

        Tensor {
            id,

            shape,
            offset: self.offset,
            len: self.len,

            data: self.data.clone(),
        }
    }

    #[inline]
    pub(super) fn as_data_slice(&self) -> TensorDataSlice<T> {
        TensorDataSlice::new(&self.data, self.offset, self.len)
    }

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
    pub fn iter_row(&self) -> slice::ChunksExact<T> {
        let dim = self.cols();

        if self.rank() > 1 {
            self.as_slice().chunks_exact(dim).into_iter()
        } else {
            self.as_slice().chunks_exact(1).into_iter()
        }
    }
}

impl<T: Clone + 'static> Tensor<T> {
    pub fn init<F>(shape: impl Into<Shape>, f: F) -> Self
    where
        F: FnMut() -> T
    {
        let shape = shape.into();
        let size = shape.size();

        let data = unsafe {
            let mut uninit = TensorUninit::<T>::new(size);

            uninit.init(f);

            uninit.into()
        };

        Self::from_data(data, shape)
    }

    pub fn fill(shape: impl Into<Shape>, value: T) -> Self {
        Self::init(shape, || value.clone())
    }

    pub fn slice<S: TensorSlice>(&self, index: S) -> Tensor<T> {
        S::slice(index, &self)
    }

    pub fn map<U, F>(&self, f: F) -> Tensor<U>
    where
        U: Clone + 'static,
        F: FnMut(&T) -> U
    {
        let data = self.as_data_slice().map(f);

        Tensor::from_data(data, self.shape())
    }

    pub fn map2<U, F, V: Clone + 'static>(
        &self, 
        rhs: &Tensor<U>,
        f: F
    ) -> Tensor<V>
    where
        V: Clone + 'static,
        F: FnMut(&T, &U) -> V
    {
        let shape = self.shape().broadcast_to(rhs.shape());

        let a = self.as_data_slice();
        let b = rhs.as_data_slice();

        let data = a.map2(&b, f);

        Tensor::from_data(data, shape)
    }

    pub fn map_slice<const M: usize, U: Clone + 'static>(
        &self, 
        f: impl Fn(&[T]) -> [U; M]
    ) -> Tensor<U> {
        let data = self.as_data_slice().map_slice(self.cols(), f);

        Tensor::from_data(data, self.shape().with_col(M))
    }
}

impl<T: Clone + Default + 'static> Tensor<T> {
    pub fn init_default(shape: impl Into<Shape>) -> Self {
        Self::init(shape, || T::default())
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

fn fmt_tensor_rec<T: fmt::Debug>(
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

impl<T: Dtype> From<()> for Tensor<T> {
    // TODO: possible conflict with Tensors
    fn from(_value: ()) -> Self {
        Tensor::empty()
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

// vec conversions

impl<T: Dtype> From<Vec<T>> for Tensor<T> {
    fn from(value: Vec<T>) -> Self {
        let len = value.len();

        Tensor::<T>::from_vec(value, Shape::from(len))
    }
}

impl<T: Dtype> From<&Vec<T>> for Tensor<T> {
    fn from(value: &Vec<T>) -> Self {
        let len = value.len();

        unsafe {
            let mut uninit = TensorUninit::<T>::new(value.len());

            let o = uninit.as_mut_slice();
            for (i, item) in value.iter().enumerate() {
                o[i] = item.clone();
            }

            uninit.into_tensor(len)
        }
    }
}

impl<T: Dtype, const N: usize> From<Vec<[T; N]>> for Tensor<T> {
    fn from(value: Vec<[T; N]>) -> Self {
        let len = value.len();

        let data = TensorData::<T>::from_vec_rows(value);
        
        Tensor::from_data(data, [len, N])
    }
}

impl<T: Dtype, const N: usize> From<&Vec<[T; N]>> for Tensor<T> {
    fn from(value: &Vec<[T; N]>) -> Self {
        let len = value.len();
        let size = len * N;

        unsafe {
            let mut uninit = TensorUninit::<T>::new(size);

            let o = uninit.as_mut_slice();
            for (j, row) in value.iter().enumerate() {
                for (i,  item) in row.iter().enumerate() {
                    o[j * N + i] = item.clone();
                }
            }

            uninit.into_tensor([len, N])
        }
    }
}

// array conversions

impl<T: Dtype, const N: usize> From<[T; N]> for Tensor<T> {
    fn from(value: [T; N]) -> Self {
        let vec = Vec::<T>::from(value);
        let len = vec.len();

        Tensor::from_vec(vec, Shape::from(len))
    }
}

impl<T: Dtype> From<&[T]> for Tensor<T> {
    fn from(value: &[T]) -> Self {
        let len = value.len();
        let size = len;

        unsafe {
            let mut uninit = TensorUninit::<T>::new(size);

            let o = uninit.as_mut_slice();
            for (i, item) in value.iter().enumerate() {
                o[i] = item.clone();
            }

            uninit.into_tensor([len])
        }
    }
}

impl<T: Dtype, const N: usize> From<&[[T; N]]> for Tensor<T> {
    fn from(value: &[[T; N]]) -> Self {
        let len = value.len();
        let size = len * N;

        unsafe {
            let mut uninit = TensorUninit::<T>::new(size);

            let o = uninit.as_mut_slice();
            for (j, row) in value.iter().enumerate() {
                for (i,  item) in row.iter().enumerate() {
                    o[j * N + i] = item.clone();
                }
            }

            uninit.into_tensor([len, N])
        }
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
    T: Dtype
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

impl<T: Dtype> FromIterator<T> for Tensor<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let vec : Vec<T> = Vec::from_iter(iter);
        let len = vec.len();

        Tensor::from_vec(vec, len)
    }
}

impl<T: Dtype + Copy + 'static> From<&Vec<Tensor<T>>> for Tensor<T> {
    fn from(values: &Vec<Tensor<T>>) -> Self {
        Tensor::<T>::from(values.as_slice())
    }
}

impl<T: Dtype + Copy + 'static> From<&[Tensor<T>]> for Tensor<T> {
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

            let o = data.as_mut_slice();
            for j in 0..n {
                let tensor = &values[j];

                for (i, value) in tensor.as_slice().iter().enumerate() {
                    o[j * sublen + i] = value.clone();
                }
            }

            Tensor::from_uninit(data, shape.push(n))
        }
    }
}

impl<T: Dtype + Copy + 'static, const N: usize> From<[Tensor<T>; N]> for Tensor<T> {
    fn from(values: [Tensor<T>; N]) -> Self {
        let vec = Vec::from(values);

        Tensor::from(&vec)
    }
}

pub trait IntoTensorList<D: Dtype> {
    fn into_list(self, vec: &mut Vec<Tensor<D>>);
}

impl<D: Dtype> IntoTensorList<D> for Vec<Tensor<D>> {
    fn into_list(self, vec: &mut Vec<Tensor<D>>) {
        let mut this = self;

        vec.append(&mut this)
    }
}

impl<D: Dtype> IntoTensorList<D> for &[Tensor<D>] {
    fn into_list(self, vec: &mut Vec<Tensor<D>>) {
        let mut vec2 = Vec::from(self);
        vec.append(&mut vec2);
    }
}

impl<D: Dtype, const N: usize> IntoTensorList<D> for [Tensor<D>; N] {
    fn into_list(self, vec: &mut Vec<Tensor<D>>) {
        let mut vec2 = Vec::from(self);
        vec.append(&mut vec2);
    }
}

macro_rules! tensor_list {
    ($($id:ident),*) => {
        #[allow(non_snake_case)]
        impl<D: Dtype, $($id),*> IntoTensorList<D> for ($($id,)*) 
        where $(
            $id: Into<Tensor<D>>
        ),*
        {
            fn into_list(self, vec: &mut Vec<Tensor<D>>) {
                let ($($id,)*) = self;

                $(
                    vec.push($id.into())
                );*
            }
        }
    }
}

tensor_list!(P0);
tensor_list!(P0, P1);
tensor_list!(P0, P1, P2);
tensor_list!(P0, P1, P2, P3);
tensor_list!(P0, P1, P2, P3, P4);
tensor_list!(P0, P1, P2, P3, P4, P5);
tensor_list!(P0, P1, P2, P3, P4, P5, P6);
tensor_list!(P0, P1, P2, P3, P4, P5, P6, P7);
tensor_list!(P0, P1, P2, P3, P4, P5, P6, P7, P8);
tensor_list!(P0, P1, P2, P3, P4, P5, P6, P7, P8, P9);

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

pub trait Dtype : Clone + Send + Sync + fmt::Debug + 'static {
    #[inline]
    fn node_op<Op>(_args: &[&Tensor<Self>], _op: &Op) -> TensorId
    where
        Op: GradOperation<Self> + Clone
    {
        TensorId::unset()
        
        // let node = NodeOp::new(&[&a, &b], binop.to_op());
    }

    fn set_tape(tensor: Tensor<Self>) -> Tensor<Self> {
        tensor // Tape::set_tensor(tensor)
    }

}

//trait Dtype : Copy {}
impl Dtype for bool {}

impl Dtype for u8 {}
impl Dtype for u16 {}
impl Dtype for u32 {}
impl Dtype for u64 {}
impl Dtype for usize {}

impl Dtype for i8 {}
impl Dtype for i16 {}
impl Dtype for i32 {}
impl Dtype for i64 {}
impl Dtype for isize {}

impl Dtype for String {}

pub type C32 = Complex<f32>;
pub type C64 = Complex<f64>;

impl Dtype for C32 {}
impl Dtype for C64 {}

impl Dtype for f32 {
    fn node_op<Op>(args: &[&Tensor<Self>], op: &Op) -> TensorId
    where
        Op: GradOperation<Self> + Clone
    {
        NodeOp::new(args, Box::new(op.clone()))
    }

    fn set_tape(tensor: Tensor<Self>) -> Tensor<Self> {
        Tape::set_tensor(tensor)
    }
}

#[cfg(test)]
mod test {
    use tensor::Shape;

    use crate::{tc32, tensor, tensor::C32, tf32};

    use super::Tensor;

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

    //
    // from vec 
    //

    #[test]
    fn tensor_from_vec() {
        let t0 = Tensor::from(vec![10, 11, 12]);
        assert_eq!(t0.len(), 3);
        assert_eq!(t0.shape().as_slice(), &[3]);
        assert_eq!(t0.as_slice(), &[10, 11, 12]);
    }

    #[test]
    fn tensor_from_vec_ref() {
        let t0 = Tensor::from(&vec![10, 11, 12]);
        assert_eq!(t0.len(), 3);
        assert_eq!(t0.shape().as_slice(), &[3]);
        assert_eq!(t0.as_slice(), &[10, 11, 12]);
    }

    #[test]
    fn tensor_from_vec_rows() {
        let t0 = Tensor::from(vec![[10, 20, 30], [11, 21, 31]]);

        assert_eq!(t0.len(), 6);
        assert_eq!(t0.cols(), 3);
        assert_eq!(t0.rows(), 2);
        assert_eq!(t0.shape().as_slice(), &[2, 3]);
        assert_eq!(t0.as_slice(), &[10, 20, 30, 11, 21, 31]);
    }

    #[test]
    fn tensor_from_vec_rows_ref() {
        let t0 = Tensor::from(&vec![[10, 20, 30], [11, 21, 31]]);

        assert_eq!(t0.len(), 6);
        assert_eq!(t0.cols(), 3);
        assert_eq!(t0.rows(), 2);
        assert_eq!(t0.shape().as_slice(), &[2, 3]);
        assert_eq!(t0.as_slice(), &[10, 20, 30, 11, 21, 31]);
    }

    #[test]
    fn tensor_init_zero() {
        let t0 = Tensor::init([4], || 0.);

        assert_eq!(t0.len(), 4);
        assert_eq!(t0.cols(), 4);
        assert_eq!(t0.rows(), 0);
        assert_eq!(t0.shape().as_slice(), &[4]);
        assert_eq!(t0, tf32!([0., 0., 0., 0.]));

        let t0 = Tensor::init([3, 2], || 0.);

        assert_eq!(t0.len(), 6);
        assert_eq!(t0.cols(), 2);
        assert_eq!(t0.rows(), 3);
        assert_eq!(t0.shape().as_slice(), &[3, 2]);
        assert_eq!(t0, tf32!([[0., 0.], [0., 0.], [0., 0.]]));
    }

    #[test]
    fn tensor_init_count() {
        let mut count = 0;
        let t0 = Tensor::init([4], || {
            let value = count;
            count += 1;
            value
        });

        assert_eq!(t0.len(), 4);
        assert_eq!(t0.cols(), 4);
        assert_eq!(t0.rows(), 0);
        assert_eq!(t0.shape().as_slice(), &[4]);
        assert_eq!(t0, tensor!([0, 1, 2, 3]));

        let mut count = 0;
        let t0 = Tensor::init([3, 2], || {
            let value = count;
            count += 1;
            value
        });

        assert_eq!(t0.len(), 6);
        assert_eq!(t0.cols(), 2);
        assert_eq!(t0.rows(), 3);
        assert_eq!(t0.shape().as_slice(), &[3, 2]);
        assert_eq!(t0, tensor!([[0, 1], [2, 3], [4, 5]]));
    }

    #[test]
    fn tensor_fill() {
        let t0 = Tensor::fill([4], 0);

        assert_eq!(t0.len(), 4);
        assert_eq!(t0.cols(), 4);
        assert_eq!(t0.rows(), 0);
        assert_eq!(t0.shape().as_slice(), &[4]);
        assert_eq!(t0, tensor!([0, 0, 0, 0]));

        let t0 = Tensor::fill([3, 2], 0.);

        assert_eq!(t0.len(), 6);
        assert_eq!(t0.cols(), 2);
        assert_eq!(t0.rows(), 3);
        assert_eq!(t0.shape().as_slice(), &[3, 2]);
        assert_eq!(t0, tensor!([[0., 0.], [0., 0.], [0., 0.]]));
    }

    #[test]
    fn tensor_zeros() {
        let t0 = Tensor::zeros([4]);

        assert_eq!(t0.len(), 4);
        assert_eq!(t0.cols(), 4);
        assert_eq!(t0.rows(), 0);
        assert_eq!(t0.shape().as_slice(), &[4]);
        assert_eq!(t0, tensor!([0, 0, 0, 0]));

        let t0 = Tensor::zeros([3, 2]);

        assert_eq!(t0.len(), 6);
        assert_eq!(t0.cols(), 2);
        assert_eq!(t0.rows(), 3);
        assert_eq!(t0.shape().as_slice(), &[3, 2]);
        assert_eq!(t0, tensor!([[0., 0.], [0., 0.], [0., 0.]]));
    }

    //
    // slices (arrays)
    //

    // Array: [Dtype]
    #[test]
    fn tensor_from_array_1d() {
        let t0 = Tensor::from([10, 11, 12]);
        assert_eq!(t0.len(), 3);
        assert_eq!(t0.shape().as_slice(), &[3]);
        assert_eq!(t0.as_slice(), &[10, 11, 12]);
    }

    // Array: &[Dtype]
    #[test]
    fn tensor_from_array_1d_ref() {
        let t0 = Tensor::from(vec![10, 11, 12].as_slice());
        assert_eq!(t0.len(), 3);
        assert_eq!(t0.shape().as_slice(), &[3]);
        assert_eq!(t0.as_slice(), &[10, 11, 12]);
    }

    // Array: [[Dtype; N]]
    #[test]
    fn tensor_from_array_2d() {
        let t0 = Tensor::from([[10, 11], [110, 111], [210, 211]]);
        assert_eq!(t0.len(), 6);
        assert_eq!(t0.shape().as_slice(), &[3, 2]);
        assert_eq!(t0.as_slice(), &[10, 11, 110, 111, 210, 211]);
    }

    // Array: &[[Dtype; N]]
    #[test]
    fn tensor_from_array_2d_ref() {
        let vec = vec![
            [10, 11], [110, 111], [210, 211]
        ];

        let t0 = Tensor::from(vec.as_slice());

        assert_eq!(t0.len(), 6);
        assert_eq!(t0.shape().as_slice(), &[3, 2]);
        assert_eq!(t0.as_slice(), &[10, 11, 110, 111, 210, 211]);
    }

    // complex

    #[test]
    fn c32_from_macro_scalar() {
        let t0 = tc32!(1., 10.);
        
        assert_eq!(t0.len(), 1);
        assert_eq!(t0.shape().as_slice(), &[]);
        assert_eq!(t0.as_slice(), &[C32 { re: 1., im: 10. }]);
    }

    #[test]
    fn c32_from_macro_vec() {
        let t0 = tc32!([(100., 10.), (101., 11.), (102., 12.)]);
        
        assert_eq!(t0.len(), 3);
        assert_eq!(t0.shape().as_slice(), &[3]);
        assert_eq!(t0.as_slice(), &[
            C32 { re: 100., im: 10. },
            C32 { re: 101., im: 11. },
            C32 { re: 102., im: 12. },
        ]);
    }

    #[test]
    fn c32_from_macro_mat() {
        let t0 = tc32!([
            [(100., 10.), (101., 11.)],
            [(200., 20.), (201., 21.)],
            [(300., 30.), (301., 31.)],
        ]);
        
        assert_eq!(t0.len(), 6);
        assert_eq!(t0.shape().as_slice(), &[3, 2]);
        assert_eq!(t0.as_slice(), &[
            C32 { re: 100., im: 10. },  C32 { re: 101., im: 11. },
            C32 { re: 200., im: 20. },  C32 { re: 201., im: 21. },
            C32 { re: 300., im: 30. },  C32 { re: 301., im: 31. }
        ]);
    }

    //
    // concatenating tensors
    //

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
        let t = Tensor::<f32>::zeros([3, 2, 4, 5]);
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
        let vec : Vec<Vec<u32>> = tensor!([1, 2, 3, 4]).iter_row().map(|v| Vec::from(v)).collect();
        let vec2 : Vec<Vec<u32>> = vec!(vec!(1), vec!(2), vec!(3), vec!(4));
        assert_eq!(vec, vec2);

        let vec : Vec<Vec<u32>> = tensor!([[1, 2], [3, 4]]).iter_row().map(|v| Vec::from(v)).collect();
        let vec2 : Vec<Vec<u32>> = vec!(vec![1, 2], vec![3, 4]);
        assert_eq!(vec, vec2);
    }

    #[test]
    fn tensor_map_i32() {
        let t1 = tensor!([1, 2, 3, 4]);
        let t2 = t1.map(|v| 2 * v);

        assert_eq!(t1.shape(), t2.shape());
        assert_eq!(t1.offset(), t2.offset());
        assert_eq!(t1.len(), t2.len());
        assert_eq!(t2, tensor!([2, 4, 6, 8]));
    }

    #[test]
    fn tensor_map_i32_to_f32() {
        let t1 = tensor!([1, 2, 3, 4]);
        let t2 = t1.map(|v| 2. * *v as f32);

        assert_eq!(t1.shape(), t2.shape());
        assert_eq!(t1.offset(), t2.offset());
        assert_eq!(t1.len(), t2.len());
        assert_eq!(t2, Tensor::from([2., 4., 6., 8.]));
    }
}