use core::fmt;
use std::{any::type_name, ops::Deref, ptr, slice, sync::Arc};

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
        let count = self.offset;
        let len = self.size;

        unsafe {
            ptr::slice_from_raw_parts(self.as_ptr().add(count), len)
                .as_ref()
                .unwrap()
        }
    }

    // Returns a possibly-wrapped pointer at the offset to support
    // broadcast
    // todo: remove because without a length, it's meaningless?
    #[inline]
    pub fn as_wrap_slice(&self, offset: usize) -> &[T] {
        let offset = if offset < self.size {
            offset
        } else {
            offset % self.size
        };

        let count = self.offset + offset;
        let len = self.size - offset;

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
            // TODO:
            //let stride : usize = shape.sublen(rank - n + 1, rank);
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

//pub trait Dtype : Clone + Send + Sync + fmt::Debug + 'static {
//}

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
    use crate::{init::fill, ten, tensor::Shape, tf32};

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
        assert_eq!(format!("{:?}", t), "Tensor<f64> {[1.0], shape: [1]}");

        let t = ten![1.];
        assert_eq!(format!("{:?}", t), "Tensor<f64> {[1.0], shape: [1]}");

        let t = ten![1., 2.];
        assert_eq!(format!("{:?}", t), "Tensor<f64> {[1.0 2.0], shape: [2]}");

        let t = ten![[1.0f32, 2., 3.], [3., 4., 5.]];
        assert_eq!(format!("{:?}", t), "Tensor<f32> {\n[[1.0 2.0 3.0],\n [3.0 4.0 5.0]], shape: [2, 3]}");

        let t = ten![
            [[1.0f32, 2.], [3., 4.]],
            [[11., 12.], [13., 14.]]
        ];
        assert_eq!(format!("{:?}", t), "Tensor<f32> {\n[[[1.0 2.0],\n [3.0 4.0]],\n\n  [[11.0 12.0],\n [13.0 14.0]]], shape: [2, 2, 2]}");

        let t = ten![
            [[1., 2.], [3., 4.]],
            [[11., 12.], [13., 14.]],
            [[21., 22.], [23., 24.]]
        ];
        assert_eq!(format!("{:?}", t), "Tensor<f64> {\n[[[1.0 2.0],\n [3.0 4.0]],\n\n  [[11.0 12.0],\n [13.0 14.0]],\n\n  [[21.0 22.0],\n [23.0 24.0]]], shape: [3, 2, 2]}");
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
        let t0 = Tensor::init_rindexed([4], |idx| idx[0]);

        assert_eq!(t0.size(), 4);
        assert_eq!(t0.cols(), 4);
        assert_eq!(t0.rows(), 0);
        assert_eq!(t0.shape().as_vec(), &[4]);
        assert_eq!(t0, ten![0, 1, 2, 3]);

        let t0 = Tensor::init_rindexed([3, 2], |idx| idx[0]);

        assert_eq!(t0.size(), 6);
        assert_eq!(t0.cols(), 2);
        assert_eq!(t0.rows(), 3);
        assert_eq!(t0.shape().as_vec(), &[3, 2]);
        assert_eq!(t0, ten![[0, 1], [0, 1], [0, 1]]);

        let t0 = Tensor::init_rindexed([3, 2], |idx| idx[1]);

        assert_eq!(t0.size(), 6);
        assert_eq!(t0.cols(), 2);
        assert_eq!(t0.rows(), 3);
        assert_eq!(t0.shape().as_vec(), &[3, 2]);
        assert_eq!(t0, ten![[0, 0], [1, 1], [2, 2]]);

        let t0 = Tensor::init_rindexed([3, 2], |idx| 
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

        let t0 = fill([3, 2], 0);

        assert_eq!(t0.size(), 6);
        assert_eq!(t0.cols(), 2);
        assert_eq!(t0.rows(), 3);
        assert_eq!(t0.shape().as_vec(), &[3, 2]);
        assert_eq!(t0, ten![[0, 0], [0, 0], [0, 0]]);
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
    fn test_flatten() {
        assert_eq!(tf32!([[1., 2.], [3., 4.]]).flatten(), tf32!([1., 2., 3., 4.]));
    }
    #[test]
    fn test_reshape() {
        assert_eq!(
            tf32!([[1., 2.], [3., 4.]]).reshape([4]),
            tf32!([1., 2., 3., 4.])
        );
    }
    #[test]
    fn test_squeeze() {
        assert_eq!(tf32!([[1.]]).squeeze(), tf32!(1.));
        assert_eq!(tf32!([[1., 2.]]).squeeze(), tf32!([1., 2.]));
        assert_eq!(tf32!([[[1.], [2.]]]).squeeze(), tf32!([1., 2.]));
    }
    
    #[test]
    fn test_squeeze_axis() {
        assert_eq!(tf32!([[[1.], [2.]]]).squeeze_axis(-1), tf32!([[1., 2.]]));
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