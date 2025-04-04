use core::fmt;
use std::{any::type_name, ops::Deref, slice, sync::Arc};

use super::{
    data::TensorData, map, slice::TensorSlice, Axis, Shape
};

pub struct Tensor<T: Type=f32> {
    shape: Shape,
    offset: usize,
    size: usize,

    data: Arc<TensorData<T>>,
}

impl<T: Type> Tensor<T> {
    pub fn empty() -> Self {
        Self {
            shape: Shape::from([0]),
            offset: 0,
            size: 0,

            data: Arc::new(TensorData::from_vec(vec![])),
        }
    }

    pub fn from_vec(vec: Vec<T>, shape: impl Into<Shape>) -> Self {
        Self {
            shape: shape.into(),
            offset: 0,
            size: vec.len(),

            data: Arc::new(TensorData::from_vec(vec)),
        }
    }
    
    pub fn from_box(data: Box<[T]>, shape: impl Into<Shape>) -> Self {
        Self {
            shape: shape.into(),
            offset: 0,
            size: data.len(),

            data: Arc::new(TensorData::from_boxed_slice(data)),
        }
    }

    pub(crate) fn from_data(data: TensorData<T>, shape: impl Into<Shape>) -> Self {
        let shape = shape.into();

        assert_eq!(shape.size(), data.len());

        Self {
            shape: shape.into(),
            offset: 0,
            size: data.len(),

            data: Arc::new(data)
        }
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.size
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
    pub fn rdim(&self, i: usize) -> usize {
        self.shape.rdim(i)
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.shape.cols()
    }

    #[inline]
    pub fn rows(&self) -> usize {
        self.shape.rows()
    }

    #[must_use]
    pub fn reshape(self, shape: impl Into<Shape>) -> Tensor<T> {
        let shape = shape.into();

        assert_eq!(shape.size(), self.size(), "shape size must match {:?} new={:?}", 
            self.shape().as_vec(), shape.as_vec()
        );

        Self { shape, ..self }
    }

    #[inline]
    pub fn get(&self, offset: usize) -> Option<&T> {
        self.data.get(self.offset + offset)
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.data.as_sub_slice(self.offset, self.size)
    }

    // Returns a possibly-wrapped pointer at the offset to support
    // broadcast
    #[inline]
    pub fn as_wrap_slice(&self, offset: usize) -> &[T] {
        let offset = if offset < self.size {
            offset
        } else {
            offset % self.size
        };

        self.data.as_sub_slice(self.offset + offset, self.size - offset)
    }

    // Returns a possibly-wrapped pointer at the offset to support
    // broadcast
    #[inline]
    pub fn as_wrap_slice_n(&self, offset: usize, len: usize) -> &[T] {
        let offset = if offset < self.size {
            offset
        } else {
            offset % self.size
        };

        self.data.as_sub_slice(self.offset + offset, len)
    }

    #[inline]
    pub unsafe fn as_ptr(&self) -> *const T {
        self.data.as_ptr().add(self.offset)
    }

    // Returns a possibly-wrapped pointer at the offset to support
    // broadcast
    #[inline]
    pub unsafe fn as_wrap_ptr(&self, offset: usize) -> *const T {
        if offset < self.size {
            self.data.as_ptr().add(self.offset + offset)
        } else {
            self.data.as_ptr().add(self.offset + offset % self.size)
        }
    }

    pub fn subslice_flat(&self, offset: usize, len: usize, shape: impl Into<Shape>) -> Self {
        assert!(offset <= self.size);
        assert!(offset + len <= self.size);

        let shape = shape.into();

        let shape_len : usize = shape.size();
        assert!(shape_len == len || shape.size() == 0 && len == 1);

        Self {
            shape,

            offset: self.offset + offset,
            size: len,

            data: self.data.clone(),
        }
    }

    // TODO: reparam to use range
    pub fn subslice(&self, offset: usize, len: usize) -> Self {
        let dim_0 = self.dim(0);

        assert!(offset <= dim_0);
        assert!(offset + len <= dim_0);

        let size : usize = self.shape().as_vec()[1..].iter().product();

        let mut shape = Vec::from(self.shape.as_vec());
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

impl<T: Type + Clone + 'static> Tensor<T> {
    pub fn from_slice(data: impl AsRef<[T]>) -> Self {
        let data = data.as_ref();
        assert!(data.len() > 0);

        Self {
            shape: Shape::from(data.len()),
            offset: 0,
            size: data.len(),

            data: Arc::new(TensorData::from_slice(data)),
        }
    }

    pub fn append(&self, tensor: impl Into<Tensor<T>>) -> Tensor<T> {
        let tensor = tensor.into();

        assert_eq!(
            self.shape().sublen(1, self.rank()), 
            tensor.shape().sublen(1, tensor.rank())
        );

        let mut vec = Vec::from(self.shape.as_vec());
        vec[0] = self.dim(0) + tensor.dim(0);

        Tensor::from_merge(&vec![self.clone(), tensor], vec)
    }

    pub fn from_merge(
        vec: &[Tensor<T>], 
        shape: impl Into<Shape>
    ) -> Self {
        let shape = shape.into();

        let len: usize = vec.iter().map(|t| t.size()).sum();

        assert_eq!(
            len, shape.size(),
            "Tensor data len={} must match shape size {:?}", len, shape.as_vec()
        );

        let mut data = Vec::<T>::new();
        data.reserve_exact(len);

        let data = vec.iter()
            .flat_map(|tensor| tensor.as_slice().iter().map(|v| v.clone()))
            .collect();

        Tensor::from_vec(data, shape)
    }
}

impl<T: Type + Clone> Tensor<T> {
    pub fn reduce(&self, f: impl FnMut(T, T) -> T) -> Tensor<T> {
        let shape = if self.shape().rank() > 1 {
            self.shape().clone().rremove(0)
        } else {
            Shape::from([1])
        };

        map::reduce(self, f).into_tensor(shape)
    }

    pub fn reduce_axis(&self, axis: impl Into<Axis>, f: impl FnMut(T, T) -> T) -> Tensor<T> {
        map::reduce_axis(self, axis, f)
    }

    pub fn init<F>(shape: impl Into<Shape>, f: F) -> Self
    where
        F: FnMut() -> T
    {
        let shape = shape.into();

        Self::from_data(map::init(&shape, f), shape)
    }

    pub fn init_indexed<F>(shape: impl Into<Shape>, f: F) -> Self
    where
        F: FnMut(&[usize]) -> T
    {
        let shape = shape.into();

        Self::from_data(map::init_indexed(&shape, f), shape)
    }

    pub fn fill(shape: impl Into<Shape>, value: T) -> Self {
        Self::init(shape, || value.clone())
    }

    pub fn slice<S: TensorSlice>(&self, index: S) -> Tensor<T> {
        S::slice(index, &self)
    }
}

impl<T: Type + Clone> Tensor<T> {
    pub fn join_vec(
        vec: &Vec<Vec<T>>, 
    ) -> Self {
        let data: Vec<T> = vec.iter()
            .flat_map(|row| row.iter().map(|v| v.clone()))
            .collect();

        let len = data.len();

        Tensor::from_vec(data, len)
    }
}

impl<T: Type + Clone + Default> Tensor<T> {
    pub fn init_default(shape: impl Into<Shape>) -> Self {
        Self::init(shape, || T::default())
    }
}

impl<T: Type> Deref for Tensor<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T: Type> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        Self { 
            shape: self.shape.clone(), 

            offset: self.offset,
            size: self.size,

            data: self.data.clone(),
        }
    }
}

impl<T: Type + PartialEq> PartialEq for Tensor<T> {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.as_slice() == other.as_slice()
    }
}

impl<'a, T: Type> IntoIterator for &'a Tensor<T> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
impl<T: Type> AsRef<Tensor<T>> for Tensor<T> {
    fn as_ref(&self) -> &Tensor<T> {
    	self
    }
}

impl<T: Type> AsRef<[T]> for Tensor<T> {
    fn as_ref(&self) -> &[T] {
	    self.as_slice()
    }
}

impl<T: Type + fmt::Debug> fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor<{}> {{", type_name::<T>())?;

        if self.shape.rank() > 1 {
            write!(f, "\n")?;
        }

        fmt_tensor_rec(&self, f, self.rank(), 0)?;
        
        write!(f, ", shape: {:?}", &self.shape.as_vec())?;
        // write!(f, ", dtype: {}", type_name::<T>())?;

        write!(f, "}}")?;
        Ok(())
    }
}

fn fmt_tensor_rec<T: Type + fmt::Debug>(
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
            let stride : usize = shape.sublen(rank - n + 1, rank);
            for j in 0..shape.rdim(n - 1) {
                if j > 0 {
                    write!(f, ",\n\n  ")?;
                }

                fmt_tensor_rec::<T>(tensor, f, n - 1, offset + j * stride)?;
            }

            write!(f, "]")
        },
    }
}

impl<T: Type> From<()> for Tensor<T> {
    // TODO: possible conflict with Tensors
    fn from(_value: ()) -> Self {
        Tensor::empty()
    }
}

impl<T: Type> From<&Tensor<T>> for Tensor<T> {
    fn from(value: &Tensor<T>) -> Self {
        value.clone()
    }
}

impl<T: Type> From<T> for Tensor<T> {
    fn from(value: T) -> Self {
        Tensor::from_vec(vec![value], Shape::scalar())
    }
}

// vec conversions

impl<T: Type> From<Vec<T>> for Tensor<T> {
    fn from(value: Vec<T>) -> Self {
        let len = value.len();

        Tensor::<T>::from_vec(value, Shape::from(len))
    }
}

impl<T: Type + Clone> From<&Vec<T>> for Tensor<T> {
    fn from(value: &Vec<T>) -> Self {
        let data = value.clone();
        let len = data.len();

        Tensor::from_vec(data, len)
    }
}

impl<T: Type, const N: usize> From<Vec<[T; N]>> for Tensor<T> {
    fn from(value: Vec<[T; N]>) -> Self {
        let len = value.len();

        let data = TensorData::<T>::from_vec_rows(value);
        
        Tensor::from_data(data, [len, N])
    }
}

impl<T: Type + Clone, const N: usize> From<&Vec<[T; N]>> for Tensor<T> {
    fn from(value: &Vec<[T; N]>) -> Self {
        let len = value.len();

        let data: Vec<T> = value.iter()
            .flat_map(|row| row.iter().map(|v| v.clone()))
            .collect();

        Tensor::from_vec(data, [len, N])
    }
}

impl<T: Type + Clone> From<&Vec<Vec<T>>> for Tensor<T> {
    fn from(value: &Vec<Vec<T>>) -> Self {
        let mut data = Vec::new();

        let len_1 = value[0].len();
        for v1 in value {
            assert_eq!(len_1, v1.len());

            for v2 in v1 {
                data.push(v2.clone());
            }
        }

        Tensor::from_vec(data, [value.len(), len_1])
    }
}

impl<T: Clone + Type> From<&Vec<Vec<Vec<T>>>> for Tensor<T> {
    fn from(value: &Vec<Vec<Vec<T>>>) -> Self {
        let mut data = Vec::new();

        let len_1 = value[0].len();
        let len_2 = value[0][0].len();

        for v1 in value {
            assert_eq!(len_1, v1.len());

            for v2 in v1 {
                assert_eq!(len_2, v2.len());

                for v3 in v2 {
                    data.push(v3.clone());
                }
            }
        }

        Tensor::from_vec(data, [value.len(), len_1, len_2])
    }
}

impl<T: Clone + Type> From<&Vec<Vec<Vec<Vec<T>>>>> for Tensor<T> {
    fn from(value: &Vec<Vec<Vec<Vec<T>>>>) -> Self {
        let mut data = Vec::new();

        let len_1 = value[0].len();
        let len_2 = value[0][0].len();
        let len_3 = value[0][0][0].len();

        for v1 in value {
            assert_eq!(len_1, v1.len());

            for v2 in v1 {
                assert_eq!(len_2, v2.len());

                for v3 in v2 {
                    assert_eq!(len_3, v3.len());

                    for v4 in v3 {
                        data.push(v4.clone());
                    }
                }
            }
        }

        Tensor::from_vec(data, [value.len(), len_1, len_2, len_3])
    }
}

impl<T: Clone + Type> From<&Vec<Vec<Vec<Vec<Vec<T>>>>>> for Tensor<T> {
    fn from(value: &Vec<Vec<Vec<Vec<Vec<T>>>>>) -> Self {
        let len_1 = value[0].len();
        let len_2 = value[0][0].len();
        let len_3 = value[0][0][0].len();
        let len_4 = value[0][0][0][0].len();

        let data: Vec<T> = value.iter().flat_map(|v1| {
            assert_eq!(len_1, v1.len());

            v1.iter().flat_map(|v2| {
                assert_eq!(len_2, v2.len());

                v2.iter().flat_map(|v3| {
                    assert_eq!(len_3, v3.len());

                    v3.iter().flat_map(|v4| {
                        assert_eq!(len_4, v4.len());

                        v4.iter().map(|v5| v5.clone())
                    })
                })
            })
        }).collect();

        Tensor::from_vec(data, [value.len(), len_1, len_2, len_3, len_4])
    }
}

// array conversions

// can't generalize from Dtype because internal array's can't match such
// as [[1., 2.]], where T=f32 not T=[f32; 2]
impl<T: Clone + Type + 'static> From<&[T]> for Tensor<T> {
    fn from(value: &[T]) -> Self {
        let len = value.len();

        let data = Vec::from(value);

        Tensor::from_vec(data, [len])
    }
}

impl<T: Type + 'static, const N: usize> From<[T; N]> for Tensor<T> {
    fn from(value: [T; N]) -> Self {
        let vec = Vec::<T>::from(value);
        let len = vec.len();

        Tensor::from_vec(vec, Shape::from(len))
    }
}

impl<T: Clone + Type, const N: usize> From<&[[T; N]]> for Tensor<T> {
    fn from(value: &[[T; N]]) -> Self {
        let len = value.len();

        let data: Vec<T> = value.iter()
            .flat_map(|row| row.iter().map(|v| v.clone()))
            .collect();

        Tensor::from_vec(data, [len, N])
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
    T: Type,
{
    fn from(value: [[T; N]; M]) -> Self {
        let data: Vec<T> = value.into_iter()
            .flatten()
            .collect();

        Tensor::from_vec(data, [M, N])
    }
}

impl<T, const N: usize, const M: usize, const L: usize>
    From<[[[T; N]; M]; L]> for Tensor<T>
where
    T: Type
{
    fn from(value: [[[T; N]; M]; L]) -> Self {
        let data: Vec<T> = value.into_iter()
            .flatten()
            .flatten()
            .collect();

        Tensor::from_vec(data, [L, M, N])
    }
}

impl<T, const N: usize, const M: usize, const L: usize, const K: usize>
    From<[[[[T; N]; M]; L]; K]> for Tensor<T>
where
    T: Type
{
    fn from(value: [[[[T; N]; M]; L]; K]) -> Self {
        let data: Vec<T> = value.into_iter()
            .flatten()
            .flatten()
            .flatten()
            .collect();

        Tensor::from_vec(data, [K, L, M, N])
    }
}

impl<T: Type> FromIterator<T> for Tensor<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let vec : Vec<T> = Vec::from_iter(iter);
        let len = vec.len();

        Tensor::from_vec(vec, len)
    }
}

impl<'a, T: Type + Clone> FromIterator<&'a T> for Tensor<T> {
    fn from_iter<I: IntoIterator<Item = &'a T>>(iter: I) -> Self {
        let vec : Vec<T> = iter.into_iter().map(|v| v.clone()).collect();
        let len = vec.len();

        Tensor::from_vec(vec, len)
    }
}

impl<T: Type + Copy + 'static> From<&Vec<Tensor<T>>> for Tensor<T> {
    fn from(values: &Vec<Tensor<T>>) -> Self {
        Tensor::<T>::from(values.as_slice())
    }
}

impl<T: Type + Copy + 'static> From<&[Tensor<T>]> for Tensor<T> {
    fn from(values: &[Tensor<T>]) -> Self {
        let n = values.len();
        assert!(n > 0);

        let sublen = values[0].size();

        let shape = values[0].shape();

        for value in values {
            assert_eq!(sublen, value.size(), "tensor length must match");
            assert_eq!(shape, value.shape(), "tensor shapes must match");
        }

        let data : Vec<T> = values.iter().flat_map(|tensor|
            tensor.as_slice().iter().map(|v| v.clone())
        ).collect();

        Tensor::from_vec(data, shape.push(n)) 
    }
}

impl<T: Type + Copy + 'static, const N: usize> From<[Tensor<T>; N]> for Tensor<T> {
    fn from(values: [Tensor<T>; N]) -> Self {
        let vec = Vec::from(values);

        Tensor::from(&vec)
    }
}

pub trait IntoTensorList<T: Type> {
    fn into_list(self, vec: &mut Vec<Tensor<T>>);
}

impl<T: Type> IntoTensorList<T> for Vec<Tensor<T>> {
    fn into_list(self, vec: &mut Vec<Tensor<T>>) {
        let mut this = self;

        vec.append(&mut this)
    }
}

impl<T: Type> IntoTensorList<T> for &[Tensor<T>] {
    fn into_list(self, vec: &mut Vec<Tensor<T>>) {
        let mut vec2 = Vec::from(self);
        vec.append(&mut vec2);
    }
}

impl<T: Type, const N: usize> IntoTensorList<T> for [Tensor<T>; N] {
    fn into_list(self, vec: &mut Vec<Tensor<T>>) {
        let mut vec2 = Vec::from(self);
        vec.append(&mut vec2);
    }
}

macro_rules! tensor_list {
    ($($id:ident),*) => {
        #[allow(non_snake_case)]
        impl<D: Type, $($id),*> IntoTensorList<D> for ($($id,)*) 
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

//pub trait Dtype : Clone + Send + Sync + fmt::Debug + 'static {
//}

pub trait Type: 'static {}

//trait Dtype : Copy {}
impl Type for bool {}

impl Type for u8 {}
impl Type for u16 {}
impl Type for u32 {}
impl Type for u64 {}
impl Type for usize {}

impl Type for i8 {}
impl Type for i16 {}
impl Type for i32 {}
impl Type for i64 {}
impl Type for isize {}

impl Type for f32 {}

impl Type for String {}

macro_rules! dtype_tuple {
    ($($id:ident),*) => {
        #[allow(non_snake_case)]
        impl<$($id: Type),*> Type for ($($id,)*) {}
    }
}

dtype_tuple!(P0, P1);
dtype_tuple!(P0, P1, P2);
dtype_tuple!(P0, P1, P2, P3);
dtype_tuple!(P0, P1, P2, P3, P4);
dtype_tuple!(P0, P1, P2, P3, P4, P5);
dtype_tuple!(P0, P1, P2, P3, P4, P5, P6);
dtype_tuple!(P0, P1, P2, P3, P4, P5, P6, P7);
dtype_tuple!(P0, P1, P2, P3, P4, P5, P6, P7, P8);
dtype_tuple!(P0, P1, P2, P3, P4, P5, P6, P7, P8, P9);

#[cfg(test)]
mod test {
    use crate::{tensor::Shape, ten, tf32};

    use super::Tensor;

    #[test]
    fn basic_tensor() {
        let t: Tensor<i32> = Tensor::from(&vec![
            vec![
                vec![101, 102, 103, 104],
                vec![111, 112, 113, 114],
                vec![121, 122, 123, 124],
            ],
            vec![
                vec![201, 202, 203, 204],
                vec![211, 212, 213, 214],
                vec![221, 222, 223, 224],
            ]
        ]);

        assert_eq!(t.size(), 2 * 3 * 4);
        assert_eq!(t.offset(), 0);
        assert_eq!(t.shape(), &[2, 3, 4].into());
        assert_eq!(t.rank(), 3);
        assert_eq!(t.dim(0), 2);
        assert_eq!(t.dim(1), 3);
        assert_eq!(t.dim(2), 4);
        assert_eq!(t.rdim(0), 4);
        assert_eq!(t.rdim(1), 3);
        assert_eq!(t.rdim(2), 2);
        assert_eq!(t.cols(), 4);
        assert_eq!(t.rows(), 3);

        let slice = [
            101, 102, 103, 104,
            111, 112, 113, 114,
            121, 122, 123, 124,
            201, 202, 203, 204,
            211, 212, 213, 214,
            221, 222, 223, 224,
        ];

        assert_eq!(t.as_slice(), &slice);

        let ten2 = Tensor::from(Vec::from(&slice));
        assert_ne!(t, ten2);
        assert_eq!(t, ten2.reshape([2, 3, 4]));

        for (i, v) in slice.iter().enumerate() {
            assert_eq!(t.get(i).unwrap(), v);
        }

        let vec: Vec<i32> = t.iter().map(|v| *v).collect();
        assert_eq!(vec, Vec::from(&slice));
    }

    #[test]
    fn debug_tensor_from_f32() {
        let t = Tensor::from(10.5);
        assert_eq!(format!("{:?}", t), "Tensor {10.5, shape: [], dtype: f32");
    }

    #[test]
    fn debug_vector_from_slice_f32() {
        let t = Tensor::from([10.5]);
        assert_eq!(format!("{:?}", t), "Tensor{[10.5], shape: [1], dtype: f32}");

        let t = Tensor::from([1., 2.]);
        assert_eq!(format!("{:?}", t), "Tensor{[1 2], shape: [2], dtype: f32}");

        let t = Tensor::from([1., 2., 3., 4., 5.]);
        assert_eq!(format!("{:?}", t), "Tensor{[1 2 3 4 5], shape: [5], dtype: f32}");
    }

    #[test]
    fn debug_matrix_from_slice_f32() {
        let t = Tensor::from([[10.5]]);
        assert_eq!(format!("{:?}", t), "Tensor{\n[[10.5]], shape: [1, 1], dtype: f32}");

        let t = Tensor::from([[1., 2.]]);
        assert_eq!(format!("{:?}", t), "Tensor{\n[[1 2]], shape: [1, 2], dtype: f32}");

        let t = Tensor::from([[1., 2., 3.], [4., 5., 6.]]);
        assert_eq!(format!("{:?}", t), "Tensor{\n[[1 2 3],\n [4 5 6]], shape: [2, 3], dtype: f32}");
    }

    #[test]
    fn debug_tensor3_from_slice_f32() {
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
        let t = ten![1.];
        assert_eq!(format!("{:?}", t), "Tensor<f32> {[1.0], shape: [1]}");

        let t = ten![1.];
        assert_eq!(format!("{:?}", t), "Tensor<f32> {[1.0], shape: [1]}");

        let t = ten![1., 2.];
        assert_eq!(format!("{:?}", t), "Tensor<f32> {[1.0 2.0], shape: [2]}");

        let t = ten![[1., 2., 3.], [3., 4., 5.]];
        assert_eq!(format!("{:?}", t), "Tensor<f32> {\n[[1.0 2.0 3.0],\n [3.0 4.0 5.0]], shape: [2, 3]}");

        let t = ten![
            [[1., 2.], [3., 4.]],
            [[11., 12.], [13., 14.]]
        ];
        assert_eq!(format!("{:?}", t), "Tensor<f32> {\n[[[1.0 2.0],\n [3.0 4.0]],\n\n  [[11.0 12.0],\n [13.0 14.0]]], shape: [2, 2, 2]}");
    }

    #[test]
    fn tensor_0_from_scalar() {
        let t0 : Tensor = 0.25.into();

        assert_eq!(t0.size(), 1);
        assert_eq!(t0.shape(), &Shape::scalar());
        assert_eq!(t0.get(0), Some(&0.25));
    }

    #[test]
    fn tensor_from_f32() {
        let t = Tensor::from(10.5);
        assert_eq!(t.size(), 1);
        assert_eq!(t[0], 10.5);
        assert_eq!(t.as_slice(), &[10.5]);
        assert_eq!(t.shape(), &Shape::scalar());
    }

    #[test]
    fn tensor_from_scalar_iterator() {
        let t0 = Tensor::from_iter(0..6);
        assert_eq!(t0.size(), 6);
        assert_eq!(t0.shape().as_vec(), &[6]);
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
        assert_eq!(t0.size(), 3);
        assert_eq!(t0.shape().as_vec(), &[3]);
        assert_eq!(t0.as_slice(), &[10, 11, 12]);
    }

    #[test]
    fn tensor_from_vec_ref() {
        let t0 = Tensor::from(&vec![10, 11, 12]);
        assert_eq!(t0.size(), 3);
        assert_eq!(t0.shape().as_vec(), &[3]);
        assert_eq!(t0.as_slice(), &[10, 11, 12]);
    }

    #[test]
    fn tensor_from_vec_rows() {
        let t0 = Tensor::from(vec![[10, 20, 30], [11, 21, 31]]);

        assert_eq!(t0.size(), 6);
        assert_eq!(t0.cols(), 3);
        assert_eq!(t0.rows(), 2);
        assert_eq!(t0.shape().as_vec(), &[2, 3]);
        assert_eq!(t0.as_slice(), &[10, 20, 30, 11, 21, 31]);
    }

    #[test]
    fn tensor_from_vec_rows_ref() {
        let t0 = Tensor::from(&vec![[10, 20, 30], [11, 21, 31]]);

        assert_eq!(t0.size(), 6);
        assert_eq!(t0.cols(), 3);
        assert_eq!(t0.rows(), 2);
        assert_eq!(t0.shape().as_vec(), &[2, 3]);
        assert_eq!(t0.as_slice(), &[10, 20, 30, 11, 21, 31]);
    }

    #[test]
    fn tensor_into() {
        assert_eq!(into([3.]), ten![3.]);
    }

    fn _as_ref(tensor: impl AsRef<Tensor>) -> Tensor {
        tensor.as_ref().clone()
    }

    fn into(tensor: impl Into<Tensor>) -> Tensor {
        tensor.into()
    }

    #[test]
    fn tensor_init_zero() {
        let t0 = Tensor::init([4], || 0.);

        assert_eq!(t0.size(), 4);
        assert_eq!(t0.cols(), 4);
        assert_eq!(t0.rows(), 0);
        assert_eq!(t0.shape().as_vec(), &[4]);
        assert_eq!(t0, tf32!([0., 0., 0., 0.]));

        let t0 = Tensor::init([3, 2], || 0.);

        assert_eq!(t0.size(), 6);
        assert_eq!(t0.cols(), 2);
        assert_eq!(t0.rows(), 3);
        assert_eq!(t0.shape().as_vec(), &[3, 2]);
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

        assert_eq!(t0.size(), 4);
        assert_eq!(t0.cols(), 4);
        assert_eq!(t0.rows(), 0);
        assert_eq!(t0.shape().as_vec(), &[4]);
        assert_eq!(t0, ten!([0, 1, 2, 3]));

        let mut count = 0;
        let t0 = Tensor::init([3, 2], || {
            let value = count;
            count += 1;
            value
        });

        assert_eq!(t0.size(), 6);
        assert_eq!(t0.cols(), 2);
        assert_eq!(t0.rows(), 3);
        assert_eq!(t0.shape().as_vec(), &[3, 2]);
        assert_eq!(t0, ten!([[0, 1], [2, 3], [4, 5]]));
    }

    #[test]
    fn tensor_init_indexed() {
        let t0 = Tensor::init_indexed([4], |idx| idx[0]);

        assert_eq!(t0.size(), 4);
        assert_eq!(t0.cols(), 4);
        assert_eq!(t0.rows(), 0);
        assert_eq!(t0.shape().as_vec(), &[4]);
        assert_eq!(t0, ten![0, 1, 2, 3]);

        let t0 = Tensor::init_indexed([3, 2], |idx| idx[0]);

        assert_eq!(t0.size(), 6);
        assert_eq!(t0.cols(), 2);
        assert_eq!(t0.rows(), 3);
        assert_eq!(t0.shape().as_vec(), &[3, 2]);
        assert_eq!(t0, ten![[0, 1], [0, 1], [0, 1]]);

        let t0 = Tensor::init_indexed([3, 2], |idx| idx[1]);

        assert_eq!(t0.size(), 6);
        assert_eq!(t0.cols(), 2);
        assert_eq!(t0.rows(), 3);
        assert_eq!(t0.shape().as_vec(), &[3, 2]);
        assert_eq!(t0, ten![[0, 0], [1, 1], [2, 2]]);

        let t0 = Tensor::init_indexed([3, 2], |idx| 
            if idx[1] == idx[0] { 1 } else { 0 }
        );

        assert_eq!(t0.size(), 6);
        assert_eq!(t0.cols(), 2);
        assert_eq!(t0.rows(), 3);
        assert_eq!(t0.shape().as_vec(), &[3, 2]);
        assert_eq!(t0, ten![[1, 0], [0, 1], [0, 0]]);
    }

    #[test]
    fn tensor_fill() {
        let t0 = Tensor::fill([4], 0);

        assert_eq!(t0.size(), 4);
        assert_eq!(t0.cols(), 4);
        assert_eq!(t0.rows(), 0);
        assert_eq!(t0.shape().as_vec(), &[4]);
        assert_eq!(t0, ten!([0, 0, 0, 0]));

        let t0 = Tensor::fill([3, 2], 0.);

        assert_eq!(t0.size(), 6);
        assert_eq!(t0.cols(), 2);
        assert_eq!(t0.rows(), 3);
        assert_eq!(t0.shape().as_vec(), &[3, 2]);
        assert_eq!(t0, ten!([[0., 0.], [0., 0.], [0., 0.]]));
    }

    #[test]
    fn tensor_zeros() {
        let t0 = Tensor::zeros([4]);

        assert_eq!(t0.size(), 4);
        assert_eq!(t0.cols(), 4);
        assert_eq!(t0.rows(), 0);
        assert_eq!(t0.shape().as_vec(), &[4]);
        assert_eq!(t0, ten!([0, 0, 0, 0]));

        let t0 = Tensor::zeros([3, 2]);

        assert_eq!(t0.size(), 6);
        assert_eq!(t0.cols(), 2);
        assert_eq!(t0.rows(), 3);
        assert_eq!(t0.shape().as_vec(), &[3, 2]);
        assert_eq!(t0, ten!([[0., 0.], [0., 0.], [0., 0.]]));
    }

    //
    // slices (arrays)
    //

    // Array: [Dtype]
    #[test]
    fn tensor_from_array_1d() {
        let t0 = Tensor::from([10, 11, 12]);
        assert_eq!(t0.size(), 3);
        assert_eq!(t0.shape().as_vec(), &[3]);
        assert_eq!(t0.as_slice(), &[10, 11, 12]);
    }

    // Array: &[Dtype]
    #[test]
    fn tensor_from_array_1d_ref() {
        let t0 = Tensor::from(vec![10, 11, 12].as_slice());
        assert_eq!(t0.size(), 3);
        assert_eq!(t0.shape().as_vec(), &[3]);
        assert_eq!(t0.as_slice(), &[10, 11, 12]);
    }

    // Array: [[Dtype; N]]
    #[test]
    fn tensor_from_array_2d() {
        let t0 = Tensor::from([[10, 11], [110, 111], [210, 211]]);
        assert_eq!(t0.size(), 6);
        assert_eq!(t0.shape().as_vec(), &[3, 2]);
        assert_eq!(t0.as_slice(), &[10, 11, 110, 111, 210, 211]);
    }

    // Array: &[[Dtype; N]]
    #[test]
    fn tensor_from_array_2d_ref() {
        let vec = vec![
            [10, 11], [110, 111], [210, 211]
        ];

        let t0 = Tensor::from(vec.as_slice());

        assert_eq!(t0.size(), 6);
        assert_eq!(t0.shape().as_vec(), &[3, 2]);
        assert_eq!(t0.as_slice(), &[10, 11, 110, 111, 210, 211]);
    }

    //
    // concatenating tensors
    //

    #[test]
    fn tensor_from_tensor_slice() {
        let t0 = Tensor::from([ten!(2.), ten!(1.), ten!(3.)]);
        assert_eq!(t0.size(), 3);
        assert_eq!(t0.shape().as_vec(), &[3]);
        assert_eq!(t0.get(0), Some(&2.));
        assert_eq!(t0.get(1), Some(&1.));
        assert_eq!(t0.get(2), Some(&3.));

        let t1 = Tensor::from([
            ten!([1., 2.]), 
            ten!([2., 3.]), 
            ten!([3., 4.])]
        );
        assert_eq!(t1.size(), 6);
        assert_eq!(t1.shape().as_vec(), &[2, 3]);
        assert_eq!(t1[0], 1.);
        assert_eq!(t1[1], 2.);
        assert_eq!(t1[2], 2.);
        assert_eq!(t1[3], 3.);
        assert_eq!(t1[4], 3.);
        assert_eq!(t1[5], 4.);
    }

    #[test]
    fn tensor_from_vec_slice() {
        let vec = vec![ten!(2.), ten!(1.), ten!(3.)];

        let t0 = Tensor::from(vec.as_slice());
        assert_eq!(t0.size(), 3);
        assert_eq!(t0.shape().as_vec(), &[3]);
        assert_eq!(t0.get(0), Some(&2.));
        assert_eq!(t0.get(1), Some(&1.));
        assert_eq!(t0.get(2), Some(&3.));

        let vec = vec![
            ten!([1., 2.]), 
            ten!([2., 3.]), 
            ten!([3., 4.])
        ];

        let ptr = vec.as_slice();
        let t1 = Tensor::from(ptr);

        assert_eq!(t1.size(), 6);
        assert_eq!(t1.shape().as_vec(), &[2, 3]);
        assert_eq!(t1[0], 1.);
        assert_eq!(t1[1], 2.);
        assert_eq!(t1[2], 2.);
        assert_eq!(t1[3], 3.);
        assert_eq!(t1[4], 3.);
        assert_eq!(t1[5], 4.);
        
        let t1 = Tensor::from(&vec);

        assert_eq!(t1.size(), 6);
        assert_eq!(t1.shape().as_vec(), &[2, 3]);
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
        assert_eq!(t.shape().as_vec(), &[3, 2, 4, 5]);
        assert_eq!(t.rank(), 4);
        assert_eq!(t.cols(), 5);
        assert_eq!(t.rows(), 4);
        assert_eq!(t.shape().batch_len(2), 6);
        assert_eq!(t.size(), 3 * 2 * 4 * 5);
    }

    #[test]
    fn tensor_macro_float() {
        let t = ten![1.];
        assert_eq!(t.shape().as_vec(), [1]);
        assert_eq!(t, Tensor::from([1.]));

        let t = ten![1., 2., 3.];
        assert_eq!(t.shape().as_vec(), [3]);
        assert_eq!(t, Tensor::from([1., 2., 3.]));
        assert_eq!(t, [1., 2., 3.].into());

        let t = ten![[1., 2., 3.], [4., 5., 6.]];
        assert_eq!(t.shape().as_vec(), [2, 3]);
        assert_eq!(t, [[1., 2., 3.], [4., 5., 6.]].into());
    }

    #[test]
    fn tensor_macro_string() {
        let t = ten!("test");
        assert_eq!(t.shape().as_vec(), &[1]);

        assert_eq!(&t[0], "test");

        let t = ten!["t1", "t2", "t3"];
        assert_eq!(t.shape().as_vec(), &[3]);

        assert_eq!(&t[0], "t1");
        assert_eq!(&t[1], "t2");
        assert_eq!(&t[2], "t3");
    }

    #[test]
    fn tensor_iter() {
        let vec : Vec<u32> = ten!([1, 2, 3, 4]).iter().map(|v| *v).collect();
        let vec2 : Vec<u32> = vec!(1, 2, 3, 4);
        assert_eq!(vec, vec2);

        let vec : Vec<u32> = ten!([[1, 2], [3, 4]]).iter().map(|v| *v).collect();
        let vec2 : Vec<u32> = vec!(1, 2, 3, 4);
        assert!(vec.iter().zip(vec2.iter()).all(|(x, y)| x == y));
    }

    #[test]
    fn tensor_iter_slice() {
        let vec : Vec<Vec<u32>> = ten!([1, 2, 3, 4]).iter_row().map(|v| Vec::from(v)).collect();
        let vec2 : Vec<Vec<u32>> = vec!(vec!(1), vec!(2), vec!(3), vec!(4));
        assert_eq!(vec, vec2);

        let vec : Vec<Vec<u32>> = ten!([[1, 2], [3, 4]]).iter_row().map(|v| Vec::from(v)).collect();
        let vec2 : Vec<Vec<u32>> = vec!(vec![1, 2], vec![3, 4]);
        assert_eq!(vec, vec2);
    }

    #[test]
    fn tensor_map_i32() {
        let t1 = ten!([1, 2, 3, 4]);
        let t2 = t1.map(|v| 2 * v);

        assert_eq!(t1.shape(), t2.shape());
        assert_eq!(t1.offset(), t2.offset());
        assert_eq!(t1.size(), t2.size());
        assert_eq!(t2, ten!([2, 4, 6, 8]));
    }

    #[test]
    fn tensor_map_i32_to_f32() {
        let t1 = ten!([1, 2, 3, 4]);
        let t2 = t1.map(|v| 2. * *v as f32);

        assert_eq!(t1.shape(), t2.shape());
        assert_eq!(t1.offset(), t2.offset());
        assert_eq!(t1.size(), t2.size());
        assert_eq!(t2, Tensor::from([2., 4., 6., 8.]));
    }

    #[test]
    fn tensor_fold_i32() {
        let t1 = ten!([1, 2, 3, 4]);
        let t2 = t1.fold(0, |s, v| s + v);

        assert_eq!(t2.shape().as_vec(), &[1]);
        assert_eq!(t2.offset(), 0);
        assert_eq!(t2.size(), 1);
        assert_eq!(t2, ten!([10]));
    }

    #[test]
    fn tensor_reduce() {
        let t1 = ten![1, 2, 3, 4];
        let t2 = t1.reduce(|s, v| s + v);
        assert_eq!(t2, ten![10]);

        assert_eq!(t2.shape().as_vec(), &[1]);
        assert_eq!(t2.offset(), 0);
        assert_eq!(t2.size(), 1);

        let t1 = ten![[1, 2], [3, 4]];
        let t2 = t1.reduce(|s, v| s + v);
        assert_eq!(t2, ten![3, 7]);

        assert_eq!(t2.shape().as_vec(), &[2]);
        assert_eq!(t2.offset(), 0);
        assert_eq!(t2.size(), 2);
    }

    #[test]
    fn test_as_ref() {
        let t1 = ten!([1, 2, 3]);

        as_ref_i32(t1);
    }

    fn as_ref_i32(tensor: impl AsRef<Tensor<i32>>) {
        let _my_ref = tensor.as_ref();
    }

    #[test]
    fn test_as_slice() {
        let t1 = ten!([1, 2, 3]);

        as_slice(t1);
    }

    fn as_slice(slice: impl AsRef<[i32]>) {
        let _my_ref = slice.as_ref();
    }

    #[test]
    fn test_collect() {
        let t1: Tensor<i32> = [1, 2, 3].iter().collect();

        assert_eq!(t1, ten!([1, 2, 3]));

        let t1: Tensor<i32> = [1, 2, 3].into_iter().collect();

        assert_eq!(t1, ten!([1, 2, 3]));
    }

    #[test]
    fn test_iter() {
        let mut vec = Vec::new();

        let t1: Tensor<i32> = [1, 2, 3].iter().collect();
        for v in &t1 {
            vec.push(*v);
        }

        assert_eq!(vec, vec![1, 2, 3]);
    }
}