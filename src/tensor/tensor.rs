use core::fmt;
use std::{any::type_name, ops::Deref, ptr, slice, sync::Arc};

use num_complex::Complex;

use super::{
    data::TensorData, unsafe_init, Shape
};

pub struct Tensor<T: Type=f32> {
    pub shape: Shape,
    pub offset: usize,
    pub size: usize,

    pub(super) data: Arc<TensorData<T>>,
}

impl<T: Type> Tensor<T> {
    pub fn from_scalar(value: T) -> Self {
        let vec = vec![value];
        TensorData::from_boxed_slice(vec.into_boxed_slice(), Shape::scalar())
    }
    
    pub fn from_vec(vec: Vec<T>, shape: impl Into<Shape>) -> Self {
        TensorData::from_boxed_slice(vec.into_boxed_slice(), shape.into())
    }
    
    pub fn from_box(slice: Box<[T]>, shape: impl Into<Shape>) -> Self {
        TensorData::from_boxed_slice(slice, shape)
    }

    pub(super) fn new(data: TensorData<T>, shape: impl Into<Shape>) -> Self {
        let shape = shape.into();

        assert_eq!(shape.size(), data.len());

        Self {
            shape,
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
    pub fn get(&self, offset: usize) -> Option<&T> {
        unsafe {
            if offset < self.size {
                let count = offset + self.offset;

                self.as_ptr().add(count).as_ref()
            } else {
                None
            }
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        let len = self.size;

        unsafe {
            ptr::slice_from_raw_parts(self.as_ptr(), len)
                .as_ref()
                .unwrap()
        }
    }

    // Returns a possibly-wrapped pointer at the offset to support
    // broadcast
    // todo: remove because without a length, it's meaningless?
    #[inline]
    pub fn as_wrap_slice(&self, offset: usize) -> &[T] {
        let count = if offset < self.size {
            offset
        } else {
            offset % self.size
        };

        let len = self.size - count;

        unsafe {
            ptr::slice_from_raw_parts(self.as_ptr().add(count), len)
                .as_ref()
                .unwrap()
        }
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

        let count = self.offset + offset;
        assert!(count + len <= self.size);

        unsafe {
            ptr::slice_from_raw_parts(self.as_ptr().add(count), len)
                .as_ref()
                .unwrap()
        }
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

        // assert!(offset <= dim_0);
        assert!(offset + len <= dim_0, "subslice end={} with shape={:?}", offset + len, self.shape().as_vec());

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

impl<T: Type + Clone> Tensor<T> {
    pub fn from_slice(data: impl AsRef<[T]>) -> Self {
        let data = data.as_ref();

        unsafe {
            unsafe_init::<T>(data.len(), data.len(), |o| {
                for (i, value) in data.iter().enumerate() {
                    o.add(i).write(value.clone());
                }
            })
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
        0 => {
            match tensor.get(offset) {
                Some(v) => { write!(f, "{:?}", v) },
                None => { write!(f, "None") }
            }
        }
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

            let stride : usize = shape.rsublen(0, rank - 1);
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

pub trait Type: 'static {}

macro_rules! tensor_types {
    ($($ty:ty)*) => {
        $(
            impl Type for $ty {}
        )*
    }
}

tensor_types!(bool);
tensor_types!(i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize);
tensor_types!(f32 f64);
tensor_types!(String);

impl<T: Type> Type for Option<T> {} 
impl<T: Type> Type for Complex<T> {} 

macro_rules! tensor_tuple {
    ($($id:ident),*) => {
        #[allow(non_snake_case)]
        impl<$($id: Type),*> Type for ($($id,)*) {}
    }
}

tensor_tuple!(P0, P1);
tensor_tuple!(P0, P1, P2);
tensor_tuple!(P0, P1, P2, P3);
tensor_tuple!(P0, P1, P2, P3, P4);
tensor_tuple!(P0, P1, P2, P3, P4, P5);
tensor_tuple!(P0, P1, P2, P3, P4, P5, P6);
tensor_tuple!(P0, P1, P2, P3, P4, P5, P6, P7);
tensor_tuple!(P0, P1, P2, P3, P4, P5, P6, P7, P8);
tensor_tuple!(P0, P1, P2, P3, P4, P5, P6, P7, P8, P9);

pub fn scalar<T: Type>(value: T) -> Tensor<T> {
    Tensor::from(value)
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

macro_rules! into_tensor_list {
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

into_tensor_list!(P0);
into_tensor_list!(P0, P1);
into_tensor_list!(P0, P1, P2);
into_tensor_list!(P0, P1, P2, P3);
into_tensor_list!(P0, P1, P2, P3, P4);
into_tensor_list!(P0, P1, P2, P3, P4, P5);
into_tensor_list!(P0, P1, P2, P3, P4, P5, P6);
into_tensor_list!(P0, P1, P2, P3, P4, P5, P6, P7);
into_tensor_list!(P0, P1, P2, P3, P4, P5, P6, P7, P8);
into_tensor_list!(P0, P1, P2, P3, P4, P5, P6, P7, P8, P9);

#[cfg(test)]
mod test {
    use crate::{ten, test::{C, T}};

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
    fn tensor_scalar() {
        let t = Tensor::from(T(10));
        assert_eq!(t.as_slice(), &[T(10)]);
        assert_eq!(t.size(), 1);
        assert_eq!(t.rank(), 0);
        assert_eq!(t.cols(), 0);
        assert_eq!(format!("{:?}", t), "Tensor<essay_tensor::test::T> {T(10), shape: []}");
    }

    #[test]
    fn tensor_vector() {
        let v: [T; 0] = [];
        let t = Tensor::from(v);
        assert_eq!(t.as_slice(), &[]);
        assert_eq!(format!("{:?}", t), "Tensor<essay_tensor::test::T> {[], shape: [0]}");

        let v: Vec<T> = vec![];
        let t = Tensor::<T>::from(v);
        assert_eq!(t.as_slice(), &[]);

        let v: Vec<C> = vec![];
        let t = Tensor::from(&v);
        assert_eq!(t.as_slice(), &[]);

        let t = Tensor::from(v.as_slice());
        assert_eq!(t.as_slice(), &[]);

        let t = Tensor::from([T(10), T(11), T(12)]);
        assert_eq!(t.as_slice(), &[T(10), T(11), T(12)]);
        assert_eq!(t, ten![T(10), T(11), T(12)]);
        assert_eq!(format!("{:?}", t), "Tensor<essay_tensor::test::T> {[T(10) T(11) T(12)], shape: [3]}");

        let t = Tensor::from(vec![T(10), T(11), T(12)]);
        assert_eq!(t, ten![T(10), T(11), T(12)]);

        let t = Tensor::from(&vec![C(10), C(11), C(12)]);
        assert_eq!(t, ten![C(10), C(11), C(12)]);

        let t = Tensor::from(vec![C(10), C(11), C(12)].as_slice());
        assert_eq!(t, ten![C(10), C(11), C(12)]);

        let t = ten![T(10), T(11), T(12)];
        assert_eq!(t, ten![T(10), T(11), T(12)]);
    }

    #[test]
    fn tensor_matrix() {
        let t = Tensor::from([[T(0), T(1), T(2)], [T(10), T(11), T(12)]]);
        assert_eq!(t.as_slice(), &[T(0), T(1), T(2), T(10), T(11), T(12)]);
        assert_eq!(t, ten![[T(0), T(1), T(2)], [T(10), T(11), T(12)]]);
        assert_eq!(format!("{:?}", t), "Tensor<essay_tensor::test::T> {\n[[T(0) T(1) T(2)],\n [T(10) T(11) T(12)]], shape: [2, 3]}");

        let t = ten![[T(0), T(1), T(2)], [T(10), T(11), T(12)]];
        assert_eq!(t.as_slice(), &[T(0), T(1), T(2), T(10), T(11), T(12)]);
        assert_eq!(t.shape().as_vec(), vec![2, 3]);

        let t = Tensor::from(vec![[T(0), T(1), T(2)], [T(10), T(11), T(12)]]);
        assert_eq!(t, ten![[T(0), T(1), T(2)], [T(10), T(11), T(12)]]);

        let t = Tensor::from(&vec![vec![C(0), C(1), C(2)], vec![C(10), C(11), C(12)]]);
        assert_eq!(t, ten![[C(0), C(1), C(2)], [C(10), C(11), C(12)]]);
    }
}