use std::{slice::SliceIndex, cmp, ops::Index};

use crate::{model::{Operation, Expr, NodeOp, Tape}, Tensor};

use super::{TensorId};


#[derive(Debug, Clone, PartialEq)]
pub struct Shape {
    dims: Vec<usize>,
}

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
    pub fn dim_rev(&self, i: usize) -> usize {
        self.dims[self.dims.len() - 1 - i]
    }

    #[inline]
    pub fn idim(&self, i: isize) -> usize {
        let i = (i + self.rank() as isize) as usize % self.rank();
        
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
            assert_eq!(
                self.dim_rev(i), b.dim_rev(i), 
                "broadcast ranks must match a={:?} b={:?}",
                self.as_slice(), b.as_slice(),
            );
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
            b.sublen(0..b.rank() - b_min)
        } else { 
            self.sublen(0..self.rank() - a_min)
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

    pub(crate) fn extend_dims(&self, units: usize) -> Shape {
        self.insert(units)
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

//
// squeeze operation
//

pub fn squeeze(x: &Tensor, axis: impl Into<AxisOpt>) -> Tensor {
    let axis = axis.into();
    let op = SqueezeOp(axis.axis);

    let node = NodeOp::new(&[x], Box::new(op.clone()));

    let tensor = op.f(&[&x], node);

    Tape::set_tensor(tensor)
}

impl Tensor {
    pub fn squeeze(x: &Tensor, axis: impl Into<AxisOpt>) -> Tensor {
        squeeze(x, axis)
    }
}

#[derive(Clone)]
pub struct SqueezeOp(Option<isize>);

impl SqueezeOp {
    fn axis(&self) -> &Option<isize> {
        &self.0
    }
}

impl Operation for SqueezeOp {
    fn f(
        &self,
        args: &[&Tensor],
        id: TensorId,
    ) -> Tensor {
        let tensor = args[0];

        let shape_slice = tensor.shape().as_slice();
        let mut vec = Vec::<usize>::new();
        match self.axis() {
            None => {
                for dim in shape_slice {
                    if *dim != 1 {
                        vec.push(*dim)
                    }
                }
            },
            Some(axis) => {
                let axis = (axis + shape_slice.len() as isize) % shape_slice.len() as isize;
                let axis = axis as usize;
                
                let mut vec = Vec::<usize>::new();
                for (i, dim) in shape_slice.iter().enumerate() {
                    if i != axis || *dim != 1 {
                        vec.push(*dim)
                    }
                }
            }
        };

        tensor.clone_with_shape(vec, id)
    }

    fn df(
        &self,
        _forward: &Expr,
        _back: &mut Expr,
        _i: usize,
        _args: &[TensorId],
        _prev: TensorId,
    ) -> TensorId {
        todo!()
    }
}

#[derive(Default)]
pub struct AxisOpt {
    axis: Option<isize>,
}

impl AxisOpt {
    pub fn axis(self, axis: isize) -> Self {
        Self { axis: Some(axis), ..self }
    }

    pub fn get_axis(self) -> Option<isize> {
        self.axis
    }

    pub(crate) fn axis_with_shape(&self, shape: &Shape) -> usize {
        match self.axis {
            Some(axis) => {
                (axis + shape.rank() as isize) as usize % shape.rank()
            },
            None => 0
        }
    }
}

pub struct Axis;

impl Axis {
    pub fn axis(axis: isize) -> AxisOpt {
        AxisOpt::default().axis(axis)
    }
}

impl From<Axis> for AxisOpt {
    fn from(_value: Axis) -> Self {
        AxisOpt::default()
    }
}

impl From<()> for AxisOpt {
    fn from(_value: ()) -> Self {
        AxisOpt::default()
    }
}

impl From<isize> for AxisOpt {
    fn from(axis: isize) -> Self {
        AxisOpt::default().axis(axis)
    }
}

#[cfg(test)]
mod test {
    use crate::{prelude::*, tensor::{squeeze, shape::Axis}};
    
    #[test]
    fn test_squeeze() {
        assert_eq!(squeeze(&tf32!([[1.]]), ()), tf32!(1.));
        assert_eq!(squeeze(&tf32!([[1., 2.]]), ()), tf32!([1., 2.]));
        assert_eq!(squeeze(&tf32!([[[1.], [2.]]]), ()), tf32!([1., 2.]));
    }
    
    #[test]
    fn test_squeeze_axis() {
        assert_eq!(squeeze(&tf32!([[[1.], [2.]]]), Axis::axis(-1)), tf32!([[1., 2.]]));
    }
}
