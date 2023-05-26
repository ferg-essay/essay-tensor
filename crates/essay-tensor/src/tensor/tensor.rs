use core::fmt;
use std::{cmp::{max, self}, any::type_name, sync::Arc, ops::{Deref, Index}, slice::SliceIndex};

use super::{data::TensorData, TensorUninit, slice::TensorSlice};

pub trait Dtype : Clone + Send + Sync + fmt::Debug + 'static {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(pub usize);

pub struct Tensor<T=f32> {
    shape: Shape,
    offset: usize,
    len: usize,

    data: Arc<TensorData<T>>,

    node_id: NodeId,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Shape {
    dims: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NodeId {
    None,
    Var(String),
    Id(TensorId),
}

impl<T: 'static> Tensor<T> {
    pub fn from_vec(vec: Vec<T>, shape: impl Into<Shape>) -> Self {
        let shape = shape.into();
        let len = vec.len();

        assert!(len > 0);

        unsafe {
            Tensor::from_data(TensorData::from_vec(vec), shape, NodeId::None)
        }
    }

    pub(crate) unsafe fn from_data(
        data: TensorData<T>, 
        shape: impl Into<Shape>, 
        node: NodeId
    ) -> Self {
        let shape = shape.into();
        let len: usize = shape.size();
        
        assert_eq!(len, data.len());
        // data.checkcast::<T>(len);

        Self {
            shape,
            offset: 0,
            len,

            data: Arc::new(data),

            node_id: node,
        }
    }
}

impl<T: Clone + 'static> Tensor<T> {
    pub fn from_slice(data: &[T]) -> Self {
        assert!(data.len() > 0);

        Self {
            shape: Shape::from(data.len()),
            offset: 0,
            len: data.len(),

            data: Arc::new(TensorData::from_slice(data)),

            node_id: NodeId::None,
        }
    }
}

impl<T: Copy + 'static> Tensor<T> {
    pub fn from_uninit(data: TensorUninit<T>, shape: impl Into<Shape>) -> Self {
        Self::from_uninit_node(data, shape, NodeId::None)
    }

    pub fn from_uninit_node(
        data: TensorUninit<T>, 
        shape: impl Into<Shape>,
        node: NodeId,
    ) -> Self {
        let shape = shape.into();
        let len: usize = max(1, shape.size());

        assert_eq!(len, data.len());        
        //data.checkcast::<T>(len);

        Self {
            shape,
            offset: 0,
            len,

            data: Arc::new(unsafe { data.init() }),

            node_id: node,
        }
    }
}

impl<T> Tensor<T> {
    pub(crate) fn with_id_node(self, id: TensorId) -> Tensor<T> {
        Tensor {
            node_id: NodeId::Id(id),

            .. self
        }
    }

    pub(crate) fn with_var_node(self, name: &str) -> Tensor<T> {
        Self {
            node_id: NodeId::Var(name.to_string()),

            .. self
        }
    }

    pub fn with_node(self, node: NodeId) -> Self {
        Self {
            node_id: node,

            .. self
        }
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

    /*
    #[inline]
    pub fn data(&self) -> &Arc<TensorData> {
        &self.data
    }
    */

    pub fn op(&self) -> &NodeId {
        &self.node_id
    }

    pub(crate) fn node_id(&self) -> &NodeId {
        &self.node_id
    }

    #[inline]
    pub fn get(&self, offset: usize) -> Option<&T> {
        unsafe { self.data.get(offset) }
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
            self.data.as_ptr().add(offset)
        } else {
            self.data.as_ptr().add(offset % self.len)
        }
    }

    pub fn subslice(&self, offset: usize, len: usize, shape: impl Into<Shape>) -> Self {
        assert!(offset <= self.len);
        assert!(offset + len <= self.len);

        let shape = shape.into();

        let shape_len : usize = shape.size();
        assert!(shape_len == len || shape.size() == 0 && len == 1);

        Self {
            shape,
            offset: self.offset + offset,
            len,
            data: self.data.clone(),
            node_id: NodeId::None,
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
            shape: self.shape.clone(), 
            len: self.len,
            offset: self.offset,

            data: self.data.clone(),

            node_id: self.node_id.clone(),
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

        if self.shape.size() > 1 {
            write!(f, "\n")?;
        }

        fmt_tensor_rec(&self, f, self.rank(), 0)?;
        
        write!(f, ", shape: {:?}", &self.shape.dims)?;
        // write!(f, ", dtype: {}", type_name::<T>())?;

        if f.alternate() {
            write!(f, ", graph: {:#?}", &self.node_id)?;
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

            for j in 0..tensor.dim(0) {
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

            let stride = shape[0];
            for j in 0..shape[1] {
                if j > 0 {
                    write!(f, ",\n ")?;
                }

                fmt_tensor_rec::<T>(tensor, f, rank - 1, offset + j * stride)?;
            }

            write!(f, "]")
        },
        rank => {
            write!(f, "[")?;

            let shape = tensor.shape();
            // TODO:
            let stride : usize = shape.sublen(0..rank - 1);
            for j in 0..shape[rank - 1] {
                if j > 0 {
                    write!(f, ",\n\n  ")?;
                }

                fmt_tensor_rec::<T>(tensor, f, rank - 1, offset + j * stride)?;
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

impl<T:Dtype> From<Vec<T>> for Tensor<T> {
    fn from(value: Vec<T>) -> Self {
        let len = value.len();

        Tensor::<T>::from_vec(value, Shape::from(len))
    }
}

impl<T:Dtype, const N:usize> From<[T; N]> for Tensor<T> {
    fn from(value: [T; N]) -> Self {
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
        assert!(len > 0);

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
/*
impl<T: 'static> From<&RawTensor> for Tensor<T> {
    fn from(raw: &RawTensor) -> Self {
        raw.data.checkcast::<T>(raw.len);

        Self {
            shape: raw.shape.clone(),
            offset: raw.offset,
            len: raw.len,

            data: Arc::clone(&raw.data),

            node_id: NodeId::None,
            marker: PhantomData,
        }
    }
}

impl<T: 'static> From<&Tensor<T>> for RawTensor {
    fn from(tensor: &Tensor<T>) -> Self {
        Self {
            shape: tensor.shape.clone(),
            offset: tensor.offset,
            len: tensor.len,

            data: Arc::clone(&tensor.data),
        }
    }
}
*/

impl Shape {
    pub fn scalar() -> Self {
        Self {
            dims: Vec::new()
        }
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.dims.iter().product()
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    #[inline]
    pub fn dim(&self, i: usize) -> usize {
        self.dims[i]
    }

    #[inline]
    pub fn dim_tail(&self) -> usize {
        let rank = self.rank();

        if rank > 0 {
            self.dims[rank - 1]
        } else {
            1
        }
    }

    #[inline]
    pub fn cols(&self) -> usize {
        let rank = self.rank();

        if rank > 0 {
            self.dims[rank - 1]
        } else {
            1
        }
    }

    #[inline]
    pub fn rows(&self) -> usize {
        let rank = self.rank();

        if rank > 1 {
            self.dims[rank - 2]
        } else {
            0
        }
    }

    #[inline]
    pub fn batch_len(&self, base_rank: usize) -> usize {
        let rank = self.rank();

        if rank > base_rank {
            self.dims[0..rank - base_rank].iter().product()
        } else {
            1
        }
    }

    pub fn broadcast(&self, b: &Shape) -> usize {
        let min_rank = cmp::min(self.rank(), b.rank());
        for i in 0..min_rank {
            assert_eq!(self.dims[i], b.dims[i], "broadcast ranks must match");
        }

        if self.rank() < b.rank() { b.size() } else { self.size() }
    }

    pub fn broadcast_min(
        &self, 
        a_min: usize, 
        b: &Shape, 
        b_min: usize
    ) -> usize {
        let min_rank = cmp::min(
            self.rank() - a_min, 
            b.rank() - b_min
        );

        for i in 0..min_rank {
            assert_eq!(self.dims[i + a_min], b.dims[i + b_min], "broadcast ranks must match");
        }

        if self.rank() - a_min < b.rank() - b_min { 
            b.dims.iter().skip(b_min).product()
        } else { 
            self.dims.iter().skip(a_min).product()
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[usize] {
        self.dims.as_slice()
    }

    pub fn push(&self, dim: usize) -> Self {
        let mut dims = self.dims.clone();

        dims.push(dim);

        Self {
            dims
        }
    }

    pub fn append(&self, tail: &[usize]) -> Self {
        let mut dims = self.dims.clone();

        for dim in tail {
            dims.push(*dim);
        }

        Self {
            dims
        }
    }

    pub fn insert(&self, dim: usize) -> Self {
        let mut dims = self.dims.clone();

        dims.insert(0, dim);

        Self {
            dims
        }
    }

    #[inline]
    pub fn sublen<I>(&self, range: I) -> usize
    where
        I: SliceIndex<[usize], Output=[usize]>
     {
        self.dims[range].iter().product()
    }

    #[inline]
    pub fn as_subslice<I>(&self, range: I) -> &[usize]
    where
        I: SliceIndex<[usize], Output=[usize]>
    {
        &self.dims[range]
    }

    #[inline]
    pub fn slice<I>(&self, range: I) -> Self
    where
        I: SliceIndex<[usize], Output=[usize]>
    {
        Self {
            dims: Vec::from(&self.dims[range])
        }
    }
}

impl From<&Shape> for Shape {
    fn from(value: &Shape) -> Self {
        value.clone()
    }
}

impl From<usize> for Shape {
    fn from(value: usize) -> Self {
        Shape {
            dims: vec![value]
        }
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Shape {
            dims: dims.to_vec(),
        }
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(dims: [usize; N]) -> Self {
        Shape {
            dims: dims.to_vec(),
        }
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Shape {
            dims: dims,
        }
    }
}

impl Index<usize> for Shape {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.dims[index]
    }
}

/*
impl<I> Index<I> for Shape 
where
    I:SliceIndex<[usize], Output=Vec<usize>>
{
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.dim(index)
    }
}
*/

//trait Dtype : Copy {}
impl Dtype for bool {}
impl Dtype for f32 {}
impl Dtype for i32 {}
impl Dtype for usize {}
impl Dtype for String {}

#[cfg(test)]
mod test {
    use crate::{tensor::{tensor::Shape}};
    use crate::tensor;

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
}