use core::fmt;
use std::{cmp::max, any::type_name, fmt::Display, rc::Rc, ops::Index};

use num_traits::Float;

use super::{data::TensorData, graph::OpGraph, TensorUninit};

pub trait Dtype : Copy + PartialEq + fmt::Debug + Display + 'static {}

pub struct Tensor<D:Dtype=f32> {
    op: Option<OpGraph>,
    shape: Vec<usize>,
    data: Rc<TensorData<D>>,
}

pub trait Op : fmt::Debug + Send + Sync + 'static {
    fn box_clone(&self) -> BoxOp;
}

pub type BoxOp = Box<dyn Op>;

pub trait IntoTensor<D:Dtype> {
    fn into_tensor(&self) -> Tensor<D>;
}

impl<D:Dtype> Tensor<D> {
    pub fn new(data: Rc<TensorData<D>>, shape: &[usize]) -> Self {
        let len: usize = max(1, shape.iter().product());
        assert_eq!(data.len(), len, "tensor data size {} doesn't match shape size {}", 
            data.len(), len);
        
        Self {
            op: None,
            shape: Vec::from(shape),
            data,
        }
    }

    pub fn new_op(
        data: Rc<TensorData<D>>, 
        shape: Vec<usize>,
        op: OpGraph,
    ) -> Self {
        let len: usize = max(1, shape.iter().product());
        assert_eq!(data.len(), len, "tensor data size {} doesn't match shape size {}", 
            data.len(), len);
        
        Self {
            op: Some(op),
            shape,
            data,
        }
    }

    pub fn set_op(self, op: OpGraph) -> Self {
        Self {
            op: Some(op),
            shape: self.shape,
            data: self.data,
        }
    }

    pub fn op(&self) -> &Option<OpGraph> {
        &self.op
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
    pub fn buffer(&self) -> &Rc<TensorData<D>> {
        &self.data
    }

    pub fn get(&self, offset: usize) -> Option<D> {
        self.data.get(offset)
    }

    pub fn next_uop(
        &self, 
        data: TensorData<D>, 
        shape: Vec<usize>,
        op: Box<dyn Op>
    ) -> Tensor<D> {
        if self.op().is_none() {
            Tensor {
                data: Rc::new(data),
                shape: shape,
                op: None,
            }
        } else {
            Tensor {
                data: Rc::new(data),
                shape: shape,
                op: Some(OpGraph::new(&[self.op()], op)),
            }
        }
    }

    pub fn next_binop(
        &self, 
        b: &Tensor<D>,
        data: TensorData<D>, 
        shape: Vec<usize>,
        op: Box<dyn Op>
    ) -> Tensor<D> {
        if self.op().is_none() && b.op().is_none() {
            Tensor {
                data: Rc::new(data),
                shape: shape,
                op: None,
            }
        } else {
            Tensor {
                data: Rc::new(data),
                shape: shape,
                op: Some(OpGraph::new(&[self.op(), b.op()], op)),
            }
        }
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }
}

impl<D:Dtype> Clone for Tensor<D> {
    fn clone(&self) -> Self {
        Self { 
            op: self.op.clone(),
            shape: self.shape.clone(), 
            data: self.data.clone(),
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
        write!(f, "Tensor{{")?;

        if self.shape.len() > 1 {
            write!(f, "\n")?;
        }

        fmt_tensor_rec(&self, f, self.rank(), 0)?;
        
        write!(f, ", shape: {:?}", &self.shape)?;
        write!(f, ", dtype: {}", type_name::<D>())?;

        if f.alternate() {
            if let Some(op) = &self.op {
                write!(f, ", op: {:#?}", op)?;
            }
        }

        write!(f, "}}")?;
        Ok(())
    }
}

fn fmt_tensor_rec<D:Dtype>(
    tensor: &Tensor<D>, 
    f: &mut fmt::Formatter<'_>, 
    i: usize,
    offset: usize
) -> fmt::Result {
    match i {
        0 => write!(f, "{}", tensor.get(offset).unwrap()),
        1 => {
            write!(f, "[")?;

            let shape = tensor.shape();

            for j in 0..shape[0] {
                if j > 0 {
                    write!(f, " ")?;
                }

                fmt_tensor_rec(tensor, f, i - 1, offset + j)?;
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

                fmt_tensor_rec::<D>(tensor, f, i - 1, offset + j * stride)?;
            }

            write!(f, "]")
        },
        i => {
            write!(f, "[")?;

            let shape = tensor.shape();
            // TODO:
            let stride : usize = shape[0..i].iter().product();
            for j in 0..shape[i] {
                if j > 0 {
                    write!(f, ",\n\n  ")?;
                }

                fmt_tensor_rec::<D>(tensor, f, i - 1, offset + j * stride)?;
            }

            write!(f, "]")
        },
    }
}

impl<D:Dtype> From<D> for Tensor<D> {
    fn from(value: D) -> Self {
        unsafe {
            let mut data = TensorUninit::<D>::new(1);
            data[0] = value;

            Self {
                op: None,
                shape: Vec::new(),
                data: Rc::new(data.init()),
            }
        }
    }
}

impl From<Tensor<f32>> for f32 {
    fn from(value: Tensor) -> Self {
        value.buffer()[0]
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
                op: None,
                shape: vec!(N),
                data: Rc::new(data.init()),
            }
        }
    }
}

impl<D:Dtype, const N: usize, const M: usize> From<[[D; N]; M]> for Tensor<D> {
    fn from(value: [[D; N]; M]) -> Self {
        unsafe {
            let mut data = TensorUninit::<D>::new(N * M);

            for (j, value) in value.iter().enumerate() {
                for (i, value) in value.iter().enumerate() {
                    data[j * N + i] = *value;
                }
            }

            Self {
                op: None,
                shape: vec!(N, M),
                data: Rc::new(data.init()),
            }
        }
    }
}

impl<D:Dtype, const N: usize, const M: usize, const L: usize>
    From<[[[D; N]; M]; L]> for Tensor<D> {
    fn from(value: [[[D; N]; M]; L]) -> Self {
        unsafe {
            let mut data = TensorUninit::<D>::new(L * M * N);

            for (l, value) in value.iter().enumerate() {
                for (m, value) in value.iter().enumerate() {
                    for (n, value) in value.iter().enumerate() {
                      data[l * M * N + m * N + n] = *value;
                    }
                }
            }

            Self {
                op: None,
                shape: vec!(N, M, L),
                data: Rc::new(data.init()),
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
                op: None,
                shape: vec!(N, M, L, K),
                data: Rc::new(data.init()),
            }
        }
    }
}

// impl Dtype for f32 {}
impl<T> Dtype for T 
where
    T:Float + fmt::Display + fmt::Debug + 'static
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
        assert_eq!(format!("{:?}", t), "Tensor{1, shape: [], dtype: f32}");

        let t = tensor!([1., 2.]);
        assert_eq!(format!("{:?}", t), "Tensor{[1 2], shape: [2], dtype: f32}");

        let t = tensor!([[1., 2., 3.], [3., 4., 5.]]);
        assert_eq!(format!("{:?}", t), "Tensor{\n[[1 2 3],\n [3 4 5]], shape: [2, 3], dtype: f32}");

        let t = tensor!([
            [[1., 2.], [3., 4.]],
            [[11., 12.], [13., 14.]]
        ]);
        assert_eq!(format!("{:?}", t), "Tensor{\n[[[1 2],\n [3 4]],\n\n  [[11 12],\n [13 14]]], shape: [2, 2, 2], dtype: f32}");
    }

    #[test]
    fn tensor_0_from_scalar() {
        let t0 : Tensor = 0.3.into();
        let v0 : f32 = t0.into();
    }
}