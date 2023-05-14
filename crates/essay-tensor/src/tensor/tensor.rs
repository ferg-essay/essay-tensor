use core::fmt;
use std::{cmp::{max, self}, any::type_name, sync::Arc, ops::{Deref}, marker::PhantomData};

use super::{data::TensorData, TensorUninit, slice::TensorSlice};

pub trait Dtype : Clone + 'static {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(pub usize);

#[derive(Debug, Clone, PartialEq)]
pub enum NodeId {
    None,
    Var(String),
    Id(TensorId),
}

pub struct Tensor<T=f32> {
    shape: Vec<usize>,
    offset: usize,
    len: usize,

    data: Arc<TensorData>,

    node_id: NodeId,

    marker: PhantomData<T>,
}

pub struct RawTensor {
    shape: Vec<usize>,
    offset: usize,
    len: usize,

    data: Arc<TensorData>,
}

pub trait IntoTensor<T> {
    fn into_tensor(&self) -> Tensor<T>;
}

impl<T: 'static> Tensor<T> {
    pub fn from_vec(vec: Vec<T>, shape: &[usize]) -> Self {
        let len = vec.len();

        assert!(len > 0);

        unsafe {
            Tensor::from_data(TensorData::from_vec(vec), shape, NodeId::None)
        }
    }

    pub(crate) unsafe fn from_data(
        data: TensorData, 
        shape: &[usize], 
        node: NodeId
    ) -> Self {
        let len: usize = max(1, shape.iter().product());
        
        data.checkcast::<T>(len);

        Self {
            shape: Vec::from(shape),
            offset: 0,
            len: len,

            data: Arc::new(data),

            node_id: node,
            marker: PhantomData,
        }
    }
}

impl<T: Clone + 'static> Tensor<T> {
    pub fn from_slice(data: &[T]) -> Self {
        assert!(data.len() > 0);

        Self {
            shape: vec![data.len()],
            offset: 0,
            len: data.len(),

            data: Arc::new(TensorData::from_slice(data)),

            node_id: NodeId::None,
            marker: PhantomData,
        }
    }
}

impl<T: Copy + 'static> Tensor<T> {
    pub fn from_uninit(data: TensorUninit<T>, shape: &[usize]) -> Self {
        Self::from_uninit_node(data, shape, NodeId::None)
    }

    pub fn from_uninit_node(
        data: TensorUninit<T>, 
        shape: &[usize],
        node: NodeId,
    ) -> Self {
        let len: usize = max(1, shape.iter().product());
        
        //data.checkcast::<T>(len);

        Self {
            shape: Vec::from(shape),
            offset: 0,
            len,

            data: Arc::new(unsafe { data.init() }),

            node_id: node,
            marker: PhantomData,
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
    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    #[inline]
    pub fn dim(&self, i: usize) -> usize {
        self.shape[i]
    }

    #[inline]
    pub fn dim_tail(&self) -> usize {
        let len = self.shape.len();

        if len > 0 {
            self.shape[len - 1]
        } else {
            1
        }
    }

    #[inline]
    pub fn cols(&self) -> usize {
        let len = self.shape.len();

        if len > 0 {
            self.shape[len - 1]
        } else {
            1
        }
    }

    #[inline]
    pub fn rows(&self) -> usize {
        let len = self.shape.len();

        if len > 1 {
            self.shape[len - 2]
        } else {
            0
        }
    }

    #[inline]
    pub fn batch_len(&self, base_rank: usize) -> usize {
        let len = self.shape.len();

        if len > base_rank {
            self.shape[0..len - base_rank].iter().product()
        } else {
            1
        }
    }

    #[inline]
    pub fn broadcast(&self, b: &Self) -> usize {
        let a_shape = self.shape();
        let b_shape = b.shape();
        let min_rank = cmp::min(a_shape.len(), b.shape.len());
        for i in 0..min_rank {
            assert_eq!(a_shape[i], b_shape[i], "broadcast ranks must match");
        }

        if a_shape.len() < b_shape.len() { b.len() } else { self.len() }
    }

    #[inline]
    pub fn broadcast_min(
        &self, 
        a_min: usize, 
        b: &Self, 
        b_min: usize
    ) -> usize {
        let a_shape = self.shape();
        let b_shape = b.shape();
        let min_rank = cmp::min(
            a_shape.len() - a_min, 
            b.shape.len() - b_min
        );

        for i in 0..min_rank {
            assert_eq!(a_shape[i + a_min], b_shape[i + b_min], "broadcast ranks must match");
        }

        if a_shape.len() - a_min < b_shape.len() - b_min { 
            b_shape.iter().skip(b_min).product()
        } else { 
            a_shape.iter().skip(a_min).product()
        }
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
        unsafe { self.data.get::<T>(offset) }
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { self.data.as_sub_slice(self.offset, self.len) }
    }

    // Returns a possibly-wrapped pointer at the offset to support
    // broadcast
    #[inline]
    pub unsafe fn as_wrap_slice(&self, offset: usize) -> *const T {
        if offset < self.len {
            self.data.as_ptr::<T>().add(self.offset + offset)
        } else {
            self.data.as_ptr::<T>().add(self.offset + offset % self.len)
        }
    }

    #[inline]
    pub unsafe fn as_ptr(&self) -> *const T {
        self.data.as_ptr::<T>().add(self.offset)
    }

    // Returns a possibly-wrapped pointer at the offset to support
    // broadcast
    #[inline]
    pub unsafe fn as_wrap_ptr(&self, offset: usize) -> *const T {
        if offset < self.len {
            self.data.as_ptr::<T>().add(offset)
        } else {
            self.data.as_ptr::<T>().add(offset % self.len)
        }
    }

    pub fn subslice(&self, offset: usize, len: usize, shape: &[usize]) -> Self {
        assert!(offset < self.len);
        assert!(offset + len < self.len);

        let shape_len : usize = shape.iter().product();
        assert!(shape_len == len || shape.len() == 0 && len == 1);

        Self {
            shape: Vec::from(shape),
            offset: self.offset + offset,
            len,
            data: self.data.clone(),
            node_id: NodeId::None,
            marker: PhantomData,
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

impl<T:Clone> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        Self { 
            shape: self.shape.clone(), 
            len: self.len,
            offset: self.offset,

            data: self.data.clone(),

            node_id: self.node_id.clone(),
            marker: self.marker.clone(),
        }
    }
}

impl<T:PartialEq> PartialEq for Tensor<T> {
    fn eq(&self, other: &Self) -> bool {
        self.shape != other.shape && self.as_slice() == other.as_slice()
    }
}

impl<T: fmt::Debug> fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor {{")?;

        if self.shape.len() > 1 {
            write!(f, "\n")?;
        }

        fmt_tensor_rec(&self, f, self.rank(), 0)?;
        
        write!(f, ", shape: {:?}", &self.shape)?;
        write!(f, ", dtype: {}", type_name::<T>())?;

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
        0 => write!(f, "{:?}", tensor.get(offset).unwrap()),
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
            let stride : usize = shape[0..rank - 1].iter().product();
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

impl<S:Dtype + Copy> From<S> for Tensor<S> {
    fn from(value: S) -> Self {
        unsafe {
            let mut data = TensorUninit::<S>::new(1);
            data[0] = value;

            Tensor::from_data(data.init(), &Vec::new(), NodeId::None)
        }
    }
}

impl From<Tensor<f32>> for f32 {
    fn from(value: Tensor) -> Self {
        value[0]
    }
}

impl<D:Dtype> From<&Tensor<D>> for Tensor<D> {
    fn from(value: &Tensor<D>) -> Self {
        value.clone()
    }
}

impl<T: 'static> From<Vec<T>> for Tensor<T> {
    fn from(value: Vec<T>) -> Self {
        let len = value.len();

        Tensor::<T>::from_vec(value, &vec![len])
    }
}

impl<T:Dtype + 'static, const N:usize> From<[T; N]> for Tensor<T> {
    fn from(value: [T; N]) -> Self {
        let vec = Vec::<T>::from(value);
        let len = vec.len();

        Tensor::from_vec(vec, &vec![len])
    }
}

impl<T, const N: usize, const M: usize> From<[[T; N]; M]> for Tensor<T>
where
    T: Dtype + Copy + 'static,
{
    fn from(value: [[T; N]; M]) -> Self {
        unsafe {
            let mut data = TensorUninit::<T>::new(N * M);

            for (j, value) in value.iter().enumerate() {
                for (i, value) in value.iter().enumerate() {
                    data[j * N + i] = value.clone();
                }
            }

            Tensor::from_data(data.init(), &vec!(M, N), NodeId::None)
        }
    }
}

impl<T, const N: usize, const M: usize, const L: usize>
    From<[[[T; N]; M]; L]> for Tensor<T>
where
    T: Dtype + Copy + 'static
{
    fn from(value: [[[T; N]; M]; L]) -> Self {
        unsafe {
            let mut data = TensorUninit::<T>::new(L * M * N);

            for (l, value) in value.iter().enumerate() {
                for (m, value) in value.iter().enumerate() {
                    for (n, value) in value.iter().enumerate() {
                      data[l * M * N + m * N + n] = value.clone();
                    }
                }
            }

            Tensor::from_data(data.init(), &vec!(L, M, N), NodeId::None)
        }
    }
}

impl<T, const N: usize, const M: usize, const L: usize, const K: usize>
    From<[[[[T; N]; M]; L]; K]> for Tensor<T>
where
    T:Dtype + Copy
{
    fn from(value: [[[[T; N]; M]; L]; K]) -> Self {
        unsafe {
            let mut data = TensorUninit::<T>::new(K * L * M * N);

            for (k, value) in value.iter().enumerate() {
                for (l, value) in value.iter().enumerate() {
                    for (m, value) in value.iter().enumerate() {
                        for (n, value) in value.iter().enumerate() {
                            data[k * L * M * N + l * N * M + m * N + n] = *value;
                        }
                    }
                }
            }

            Tensor::from_data(data.init(), &vec!(N, M, L, K), NodeId::None)
        }
    }
}

impl<T:Dtype> FromIterator<T> for Tensor<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let vec : Vec<T> = Vec::from_iter(iter);
        let len = vec.len();
        assert!(len > 0);

        Tensor::from_vec(vec, &vec![len])
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

            let mut shape = shape.clone();
            shape.push(n);

            Tensor::from_uninit(data, &shape)
        }
    }
}

impl<T:Dtype + Copy + 'static, const N: usize> From<[Tensor<T>; N]> for Tensor<T> {
    fn from(values: [Tensor<T>; N]) -> Self {
        let vec = Vec::from(values);

        Tensor::from(&vec)
    }
}

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

//trait Dtype : Copy {}
impl Dtype for f32 {}
impl Dtype for i32 {}
impl Dtype for usize {}
impl Dtype for String {}

#[cfg(test)]
mod test {
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
        assert_eq!(t0.shape(), &[]);
        assert_eq!(t0.get(0), Some(&0.25));
    }

    #[test]
    fn tensor_from_f32() {
        let t = Tensor::from(10.5);
        assert_eq!(t.len(), 1);
        assert_eq!(t[0], 10.5);
        assert_eq!(t.as_slice(), &[10.5]);
        assert_eq!(t.shape(), &[]);
    }

    #[test]
    fn tensor_from_scalar_iterator() {
        let t0 = Tensor::from_iter(0..6);
        assert_eq!(t0.len(), 6);
        assert_eq!(t0.shape(), &[6]);
        for i in 0..6 {
            assert_eq!(t0.get(i), Some(&i));
        }
    }

    #[test]
    fn tensor_from_tensor_slice() {
        let t0 = Tensor::from([tensor!(2.), tensor!(1.), tensor!(3.)]);
        assert_eq!(t0.len(), 3);
        assert_eq!(t0.shape(), &[3]);
        assert_eq!(t0.get(0), Some(&2.));
        assert_eq!(t0.get(1), Some(&1.));
        assert_eq!(t0.get(2), Some(&3.));

        let t1 = Tensor::from([
            tensor!([1., 2.]), 
            tensor!([2., 3.]), 
            tensor!([3., 4.])]
        );
        assert_eq!(t1.len(), 6);
        assert_eq!(t1.shape(), &[2, 3]);
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
        assert_eq!(t0.shape(), &[3]);
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
        assert_eq!(t1.shape(), &[2, 3]);
        assert_eq!(t1[0], 1.);
        assert_eq!(t1[1], 2.);
        assert_eq!(t1[2], 2.);
        assert_eq!(t1[3], 3.);
        assert_eq!(t1[4], 3.);
        assert_eq!(t1[5], 4.);
        
        let t1 = Tensor::from(&vec);

        assert_eq!(t1.len(), 6);
        assert_eq!(t1.shape(), &[2, 3]);
        assert_eq!(t1[0], 1.);
        assert_eq!(t1[1], 2.);
        assert_eq!(t1[2], 2.);
        assert_eq!(t1[3], 3.);
        assert_eq!(t1[4], 3.);
        assert_eq!(t1[5], 4.);
    }

    #[test]
    fn shape_from_zeros() {
        let t = Tensor::zeros(&[3, 2, 4, 5]);
        assert_eq!(t.shape(), &[3, 2, 4, 5]);
        assert_eq!(t.rank(), 4);
        assert_eq!(t.cols(), 5);
        assert_eq!(t.rows(), 4);
        assert_eq!(t.batch_len(2), 6);
        assert_eq!(t.len(), 3 * 2 * 4 * 5);
    }
}