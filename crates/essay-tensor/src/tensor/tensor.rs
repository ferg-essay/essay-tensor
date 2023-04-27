use core::fmt;
use std::{cmp::max, any::type_name, fmt::Display, rc::Rc};

use super::buffer::TensorData;

pub trait Dtype : Copy + PartialEq + fmt::Debug + Display + 'static {}

pub struct Tensor<const N:usize,D:Dtype=f32> {
    op: Option<BoxOp>,
    shape: [usize; N],
    data: Rc<TensorData<D>>,
}

pub trait Op : fmt::Debug + Send + Sync + 'static {
    fn box_clone(&self) -> BoxOp;
}

pub type BoxOp = Box<dyn Op>;

pub trait IntoTensor<const N:usize, D:Dtype> {
    fn into_tensor(&self) -> Tensor<N, D>;
}

impl<const N:usize, D:Dtype> Tensor<N, D> {
    pub fn new(data: Rc<TensorData<D>>, shape: [usize; N]) -> Self {
        let len: usize = max(1, shape.iter().product());
        assert_eq!(data.len(), len, "tensor data size {} doesn't match shape size {}", 
            data.len(), len);
        
        Self {
            op: None,
            shape,
            data,
        }
    }

    pub fn new_op(
        data: Rc<TensorData<D>>, 
        shape: [usize; N],
        op: BoxOp,
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

    pub fn set_op(self, op: impl Op) -> Self {
        Self {
            op: Some(Box::new(op)),
            shape: self.shape,
            data: self.data,
        }
    }

    pub fn op(&self) -> &Option<Box<dyn Op>> {
        &self.op
    }

    pub fn shape(&self) -> &[usize; N] {
        &self.shape
    }

    pub fn buffer(&self) -> &Rc<TensorData<D>> {
        &self.data
    }

    pub fn buffer_mut(&mut self) -> &mut Rc<TensorData<D>> {
        &mut self.data
    }

    pub fn get(&self, offset: usize) -> Option<D> {
        self.data.get(offset)
    }
}

impl<const N:usize,D:Dtype> Clone for Tensor<N, D> {
    fn clone(&self) -> Self {
        Self { 
            op: if let Some(op) = &self.op {
                Some(op.box_clone())
            } else {
                None
            },
            shape: self.shape.clone(), 
            data: self.data.clone(),
        }
    }
}

impl<const N:usize, D:Dtype> PartialEq for Tensor<N, D> {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
    }
}

impl<const N:usize, D:Dtype> fmt::Debug for Tensor<N, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor{{")?;

        if self.shape.len() > 1 {
            write!(f, "\n")?;
        }

        fmt_tensor_rec(&self, f, self.shape.len(), 0)?;
        
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

fn fmt_tensor_rec<const N:usize, D:Dtype>(
    tensor: &Tensor<N, D>, 
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

                fmt_tensor_rec::<N, D>(tensor, f, i - 1, offset + j * stride)?;
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

                fmt_tensor_rec::<N, D>(tensor, f, i - 1, offset + j * stride)?;
            }

            write!(f, "]")
        },
    }
}

impl<D:Dtype> From<D> for Tensor<0, D> {
    fn from(value: D) -> Self {
        unsafe {
            let mut data = TensorData::<D>::new_uninit(1);
            data.set(0, value);

            Self {
                op: None,
                shape: [],
                data: Rc::new(data),
            }
        }
    }
}

impl<D:Dtype, const N:usize> From<[D; N]> for Tensor<1, D> {
    fn from(value: [D; N]) -> Self {
        unsafe {
            let mut data = TensorData::<D>::new_uninit(N);

            for (i, value) in value.iter().enumerate() {
                data.set(i, *value);
            }

            Self {
                op: None,
                shape: [N],
                data: Rc::new(data),
            }
        }
    }
}

impl<D:Dtype, const N: usize, const M: usize> From<[[D; N]; M]> for Tensor<2, D> {
    fn from(value: [[D; N]; M]) -> Self {
        unsafe {
            let mut data = TensorData::<D>::new_uninit(N * M);

            for (j, value) in value.iter().enumerate() {
                for (i, value) in value.iter().enumerate() {
                    data.set(j * N + i, *value);
                }
            }

            Self {
                op: None,
                shape: [N, M],
                data: Rc::new(data),
            }
        }
    }
}

impl<D:Dtype, const N: usize, const M: usize, const L: usize>
    From<[[[D; N]; M]; L]> for Tensor<3, D> {
    fn from(value: [[[D; N]; M]; L]) -> Self {
        unsafe {
            let mut data = TensorData::<D>::new_uninit(L * M * N);

            for (l, value) in value.iter().enumerate() {
                for (m, value) in value.iter().enumerate() {
                    for (n, value) in value.iter().enumerate() {
                      data.set(l * M * N + m * N + n, *value);
                    }
                }
            }

            Self {
                op: None,
                shape: [N, M, L],
                data: Rc::new(data),
            }
        }
    }
}

impl<D:Dtype, const N: usize, const M: usize, const L: usize, const K: usize>
    From<[[[[D; N]; M]; L]; K]> for Tensor<4, D> {
    fn from(value: [[[[D; N]; M]; L]; K]) -> Self {
        unsafe {
            let mut data = TensorData::<D>::new_uninit(K * L * M * N);

            for (k, value) in value.iter().enumerate() {
                for (l, value) in value.iter().enumerate() {
                    for (m, value) in value.iter().enumerate() {
                        for (n, value) in value.iter().enumerate() {
                            data.set(k * L * M * N + l * N * M + m * N + n, *value);
                        }
                    }
                }
            }

            Self {
                op: None,
                shape: [N, M, L, K],
                data: Rc::new(data),
            }
        }
    }
}

impl Dtype for f32 {}

#[cfg(test)]
mod test {
    use crate::tensor;

    use super::Tensor;

    #[test]
    fn debug_tensor_from_f32() {
        let t = Tensor::<0>::from(10.5);

        assert_eq!(format!("{:?}", t), "Tensor{10.5, shape: [], dtype: f32}");
    }

    #[test]
    fn debug_vector_from_slice_f32() {
        let t = Tensor::<1>::from([]);
        assert_eq!(format!("{:?}", t), "Tensor{[], shape: [0], dtype: f32}");

        let t = Tensor::<1>::from([10.5]);
        assert_eq!(format!("{:?}", t), "Tensor{[10.5], shape: [1], dtype: f32}");

        let t = Tensor::<1>::from([1., 2.]);
        assert_eq!(format!("{:?}", t), "Tensor{[1 2], shape: [2], dtype: f32}");

        let t = Tensor::<1>::from([1., 2., 3., 4., 5.]);
        assert_eq!(format!("{:?}", t), "Tensor{[1 2 3 4 5], shape: [5], dtype: f32}");
    }

    #[test]
    fn debug_matrix_from_slice_f32() {
        let t = Tensor::<2>::from([[]]);
        assert_eq!(format!("{:?}", t), "Tensor{\n[[]], shape: [1, 0], dtype: f32}");

        let t = Tensor::<2>::from([[10.5]]);
        assert_eq!(format!("{:?}", t), "Tensor{\n[[10.5]], shape: [1, 1], dtype: f32}");

        let t = Tensor::<2>::from([[1., 2.]]);
        assert_eq!(format!("{:?}", t), "Tensor{\n[[1 2]], shape: [1, 2], dtype: f32}");

        let t = Tensor::<2>::from([[1., 2., 3.], [4., 5., 6.]]);
        assert_eq!(format!("{:?}", t), "Tensor{\n[[1 2 3],\n [4 5 6]], shape: [2, 3], dtype: f32}");
    }

    #[test]
    fn debug_tensor3_from_slice_f32() {
        let t = Tensor::<3>::from([
            [[]]
        ]);
        assert_eq!(format!("{:?}", t), "Tensor{\n[[[]]], shape: [1, 1, 0], dtype: f32}");

        let t = Tensor::<3>::from([
            [[10.5]]
        ]);
        assert_eq!(format!("{:?}", t), "Tensor{\n[[[10.5]]], shape: [1, 1, 1], dtype: f32}");

        let t = Tensor::<3>::from([
            [[1., 2.]],
            [[101., 102.]]
        ]);
        assert_eq!(format!("{:?}", t), "Tensor{\n[[[1 2]],\n\n  [[101 102]]], shape: [2, 1, 2], dtype: f32}");

        let t = Tensor::<3>::from([
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
}