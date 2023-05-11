use core::fmt;
use std::{cmp::{max, self}, any::type_name, sync::Arc};

use num_traits::Float;

use crate::graph::{TensorId};

use super::{data::TensorData, TensorUninit};

pub trait Dtype : Copy + PartialEq + fmt::Debug + Sync + Send + 'static {}

#[derive(Debug, Clone, PartialEq)]
pub enum NodeId {
    None,
    Var(String),
    Id(TensorId),
}

pub struct Tensor<D:Dtype=f32> {
    shape: Vec<usize>,
    data: Arc<TensorData<D>>,

    node: NodeId,
}

pub trait IntoTensor<D:Dtype> {
    fn into_tensor(&self) -> Tensor<D>;
}

impl<D:Dtype> Tensor<D> {
    pub fn new(data: TensorData<D>, shape: &[usize]) -> Self {
        let len: usize = max(1, shape.iter().product());
        assert_eq!(data.len(), len, "tensor data size {} doesn't match shape size {}", 
            data.len(), len);
        
        Self {
            shape: Vec::from(shape),
            data: Arc::new(data),

            node: NodeId::None,
        }
    }

    pub(crate) fn with_id(&self, id: TensorId) -> Tensor<D> {
        Tensor {
            shape: self.shape.clone(),
            data: self.data.clone(),

            node: NodeId::Id(id),
        }
    }

    pub(crate) fn to_var(self, name: &str) -> Tensor<D> {
        Self {
            shape: self.shape,
            data: self.data,

            node: NodeId::Var(name.to_string()),
        }
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
    pub fn dim_zero(&self) -> usize {
        if self.shape.len() > 0 {
            self.shape[0]
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

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn data(&self) -> &Arc<TensorData<D>> {
        &self.data
    }

    pub fn get(&self, offset: usize) -> Option<D> {
        self.data.get(offset)
    }

    pub fn set_node(self, node: NodeId) -> Self {
        Self {
            shape: self.shape,
            data: self.data,

            node,
        }
    }

    pub fn op(&self) -> &NodeId {
        &self.node
    }

    pub(crate) fn node(&self) -> &NodeId {
        &self.node
    }

    pub fn new_op(
        data: TensorData<D>, 
        shape: Vec<usize>,
        node: NodeId,
    ) -> Self {
        let len: usize = max(1, shape.iter().product());
        assert_eq!(data.len(), len, "tensor data size {} doesn't match shape size {}", 
            data.len(), len);
        
        Self {
            shape,
            data: Arc::new(data),

            node,
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[D] {
        self.data.as_slice()
    }
}

impl<D:Dtype> Clone for Tensor<D> {
    fn clone(&self) -> Self {
        Self { 
            shape: self.shape.clone(), 
            data: self.data.clone(),

            node: self.node.clone(),
        }
    }
}

impl<D:Dtype> PartialEq for Tensor<D> {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
    }
}

impl<D:Dtype> fmt::Debug for Tensor<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor {{")?;

        if self.shape.len() > 1 {
            write!(f, "\n")?;
        }

        fmt_tensor_rec(&self, f, self.rank(), 0)?;
        
        write!(f, ", shape: {:?}", &self.shape)?;
        write!(f, ", dtype: {}", type_name::<D>())?;

        if f.alternate() {
            write!(f, ", graph: {:#?}", &self.node)?;
        }

        write!(f, "}}")?;
        Ok(())
    }
}

fn fmt_tensor_rec<D:Dtype>(
    tensor: &Tensor<D>, 
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

                fmt_tensor_rec::<D>(tensor, f, rank - 1, offset + j * stride)?;
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

                fmt_tensor_rec::<D>(tensor, f, rank - 1, offset + j * stride)?;
            }

            write!(f, "]")
        },
    }
}

impl From<f32> for Tensor<f32> {
    fn from(value: f32) -> Self {
        unsafe {
            let mut data = TensorUninit::<f32>::new(1);
            data[0] = value;

            Self {
                shape: Vec::new(),
                data: Arc::new(data.init()),
                node: NodeId::None,
            }
        }
    }
}

impl From<Tensor<f32>> for f32 {
    fn from(value: Tensor) -> Self {
        value.data()[0]
    }
}

impl<D:Dtype> From<&Tensor<D>> for Tensor<D> {
    fn from(value: &Tensor<D>) -> Self {
        value.clone()
    }
}

impl<const N:usize> From<[f32; N]> for Tensor<f32> {
    fn from(value: [f32; N]) -> Self {
        unsafe {
            let mut data = TensorUninit::<f32>::new(N);

            for (i, value) in value.iter().enumerate() {
                data[i] = *value;
            }

            Self {
                shape: vec!(N),
                data: Arc::new(data.init()),

                node: NodeId::None,
            }
        }
    }
}

impl<const N: usize, const M: usize> From<[[f32; N]; M]> for Tensor<f32> {
    fn from(value: [[f32; N]; M]) -> Self {
        unsafe {
            let mut data = TensorUninit::<f32>::new(N * M);

            for (j, value) in value.iter().enumerate() {
                for (i, value) in value.iter().enumerate() {
                    data[j * N + i] = *value;
                }
            }

            Self {
                shape: vec!(N, M),
                data: Arc::new(data.init()),

                node: NodeId::None,
            }
        }
    }
}

impl<const N: usize, const M: usize, const L: usize>
    From<[[[f32; N]; M]; L]> for Tensor<f32> {
    fn from(value: [[[f32; N]; M]; L]) -> Self {
        unsafe {
            let mut data = TensorUninit::<f32>::new(L * M * N);

            for (l, value) in value.iter().enumerate() {
                for (m, value) in value.iter().enumerate() {
                    for (n, value) in value.iter().enumerate() {
                      data[l * M * N + m * N + n] = *value;
                    }
                }
            }

            Self {
                shape: vec!(N, M, L),
                data: Arc::new(data.init()),

                node: NodeId::None,
            }
        }
    }
}

impl<D:Dtype, const N: usize, const M: usize, const L: usize, const K: usize>
    From<[[[[D; N]; M]; L]; K]> for Tensor<D> {
    fn from(value: [[[[D; N]; M]; L]; K]) -> Self {
        unsafe {
            let mut data = TensorUninit::<D>::new(K * L * M * N);

            for (k, value) in value.iter().enumerate() {
                for (l, value) in value.iter().enumerate() {
                    for (m, value) in value.iter().enumerate() {
                        for (n, value) in value.iter().enumerate() {
                            data[k * L * M * N + l * N * M + m * N + n] = *value;
                        }
                    }
                }
            }

            Self {
                shape: vec!(N, M, L, K),
                data: Arc::new(data.init()),

                node: NodeId::None,
            }
        }
    }
}

// impl Dtype for f32 {}
impl<T> Dtype for T 
where
    T:Float + fmt::Display + fmt::Debug + Sync + Send + 'static
{

}

#[cfg(test)]
mod test {
    use crate::tensor;

    use super::Tensor;

    #[test]
    fn debug_tensor_from_f32() {
        let t = Tensor::from(10.5);
        assert_eq!(format!("{:?}", t), "Tensor{10.5, shape: [], dtype: f32}");
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

        let t = Tensor::from([
            [[10.5]]
        ]);
        assert_eq!(format!("{:?}", t), "Tensor{\n[[[10.5]]], shape: [1, 1, 1], dtype: f32}");

        let t = Tensor::from([
            [[1., 2.]],
            [[101., 102.]]
        ]);
        assert_eq!(format!("{:?}", t), "Tensor{\n[[[1 2]],\n\n  [[101 102]]], shape: [2, 1, 2], dtype: f32}");

        let t = Tensor::from([
            [[1., 2.], [3., 4.]],
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
        let t0 : Tensor = 0.3.into();
        let _v0 : f32 = t0.into();
    }
}