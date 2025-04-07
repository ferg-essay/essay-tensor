use std::cmp;

use super::{Axis, Tensor, Type};

#[derive(Debug, Clone, PartialEq)]
pub struct Shape {
    dims: [u32; Shape::MAX_RANK],
    rank: usize,
}

impl Shape {
    pub const MAX_RANK: usize = 6;

    pub fn scalar() -> Self {
        let dims = [0; Self::MAX_RANK];

        Self {
            dims,
            rank: 0,
        }
    }

    pub fn vector(len: usize) -> Self {
        assert!(len > 0, "Shape requires a non-zero size");

        let mut dims = [0; Self::MAX_RANK];
        dims[0] = len as u32;

        Self {
            dims,
            rank: 1,
        }
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.dims[0..self.rank as usize].iter().product::<u32>() as usize
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.rank as usize
    }

    ///
    /// Returns the dimension ordered from outside-in
    /// 
    #[inline]
    pub fn dim(&self, i: usize) -> usize {
        self.dims[self.rank - 1 - i] as usize
    }

    ///
    /// Sets the dimension ordered from outside-in
    /// 
    #[inline]
    #[must_use]
    pub fn with_dim(self, i: usize, value: usize) -> Self {
        self.replace(i, value)
    }

    ///
    /// Sets the dimension
    /// 
    #[inline]
    #[must_use]
    pub fn replace(self, i: usize, value: usize) -> Self {
        let rank = self.rank;

        assert!(i < rank);

        self.rreplace(rank - 1 - i, value)
    }

    ///
    /// Sets the dimension
    /// 
    #[inline]
    #[must_use]
    pub fn rreplace(mut self, i: usize, value: usize) -> Self {
        assert!(i < self.rank);

        self.dims[i] = value as u32;

        self
    }

    ///
    /// Dimension indexed in reverse order, so 0 is cols, 1 is rows
    /// 
    #[inline]
    pub fn rdim(&self, i: usize) -> usize {
        self.dims[i] as usize
    }

    #[inline]
    pub fn idim(&self, i: isize) -> usize {
        let i = (i + self.rank() as isize) as usize % self.rank();
        
        self.dim(i)
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.dims[0] as usize
    }
    
    #[inline]
    #[must_use]
    pub fn with_cols(self, col: usize) -> Self {
        let mut own = self;

        own.dims[0] = col as u32;

        own
    }

    #[inline]
    pub fn rows(&self) -> usize {
        let rank = self.rank();

        if rank > 1 {
            self.dims[1] as usize
        } else {
            0
        }
    }
    
    #[inline]
    #[must_use]
    pub fn with_rows(self, row: usize) -> Self {
        assert!(self.rank > 1);

        let mut own = self;

        own.dims[1] = row as u32;

        own
    }

    #[inline]
    pub fn batch_len(&self, base_rank: usize) -> usize {
        let rank = self.rank();

        if base_rank < rank {
            self.dims[0..rank - base_rank].iter().product::<u32>() as usize
        } else {
            1
        }
    }

    pub fn broadcast(&self, b: &Shape) -> usize {
        let min_rank = cmp::min(self.rank(), b.rank());
        for i in 0..min_rank {
            assert_eq!(
                self.rdim(i), b.rdim(i), 
                "broadcast ranks must match a={:?} b={:?}",
                self.as_vec(), b.as_vec(),
            );
        }

        if self.rank() < b.rank() { b.size() } else { self.size() }
    }

    pub fn check_broadcast(&self, b: &Shape) {
        let min_rank = cmp::min(self.rank(), b.rank());
        for i in 0..min_rank {
            assert_eq!(
                self.rdim(i), b.rdim(i), 
                "broadcast ranks must match a={:?} b={:?}",
                self.as_vec(), b.as_vec(),
            );
        }
    }

    pub fn broadcast_to(&self, b: &Shape) -> Self {
        let min_rank = cmp::min(self.rank(), b.rank());
        for i in 0..min_rank {
            assert_eq!(
                self.rdim(i), b.rdim(i), 
                "broadcast ranks must match a={:?} b={:?}",
                self.as_vec(), b.as_vec(),
            );
        }

        if self.rank() < b.rank() {
            b.clone()
        } else {
            self.clone()
        }
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
            b.sublen(0, b.rank() - b_min)
        } else { 
            self.sublen(0, self.rank() - a_min)
        }
    }

    #[inline]
    pub fn as_vec(&self) -> Vec<usize> {
        let mut vec = Vec::new();

        for i in 0..self.rank as usize {
            vec.push(self.dim(i))
        }

        vec
    }

    pub fn push(&self, dim: usize) -> Self {
        let mut dims = self.dims.clone();
        dims[self.rank as usize] = dim as u32;

        Self {
            dims,
            rank: self.rank + 1,
        }
    }

    #[must_use]
    pub fn append(&self, tail: &[usize]) -> Self {
        let mut dims = self.dims.clone();

        for (i, dim) in tail.iter().enumerate() {
            dims[i + self.rank as usize] = *dim as u32;
        }

        Self {
            dims,
            rank: self.rank + dims.len(),
        }
    }

    #[inline]
    pub fn sublen(&self, start: usize, end: usize) -> usize
    {
        self.rsublen(self.rev(start), self.rev(end))
    }

    #[inline]
    pub fn rsublen(&self, start: usize, end: usize) -> usize
    {
        let mut size = 1;

        for i in start..end {
            size *= self.dims[i];
        }

        size as usize
    }

    fn rev(&self, value: usize) -> usize {
        self.rank as usize - 1 - value
    }

    #[inline]
    #[must_use]
    pub fn expand_dims(&self, axis: isize) -> Self {
        let mut vec = self.as_vec();

        let index = if axis >= 0 { axis } else { vec.len() as isize + axis + 1 } as usize;
        assert!(index <= vec.len(), "expand_dims axis is invalid {} in {:?}", axis, self.as_vec());

        vec.insert(index, 1);

        Self::from(vec)
    }

    #[must_use]
    pub fn squeeze(&self, axis: impl Into<Axis>) -> Self {
        let axis: Axis = axis.into();

        match axis.get_axis() {
            None => {
                let mut rank = 0;
                let mut dims = [0; Self::MAX_RANK];
                dims[0] = 1;

                for i in 0..self.rank as usize {
                    if self.dims[i] != 1 {
                        dims[rank] = self.dims[i];
                        rank += 1;
                    }
                }

                Self {
                    dims,
                    rank: rank.max(1),
                }
            },
            Some(axis) => {
                let len = self.rank;
                let axis = (axis + len as isize) % len as isize;
                let axis = self.rank as usize - axis as usize;
                
                let mut rank = 0;
                let mut dims = [0; Self::MAX_RANK];
                dims[0] = 1;

                for i in 0..self.rank as usize {
                    if i != axis || self.dims[i] != 1 {
                        dims[rank] = self.dims[i];
                        rank += 1;
                    }
                }

                Self {
                    dims,
                    rank: rank.max(1),
                }
            }
        }
    }

    pub fn pop(&self) -> Shape {
        let mut dims = self.dims.clone();

        dims[self.rank as usize - 1] = 0;

        Self {
            dims,
            rank: self.rank.max(1) - 1,
        }
    }

    #[must_use]
    pub fn remove(self, axis: usize) -> Shape {
        assert!(axis < self.rank);

        let axis = self.rank - 1 - axis;

        self.rremove(axis)
    }

    #[must_use]
    pub fn rremove(mut self, axis: usize) -> Shape {
        assert!(axis < self.rank);

        for i in axis..self.rank - 1 {
            self.dims[i] = self.dims[i + 1];
        }

        self.rank = self.rank - 1;
        self.dims[self.rank] = 0;

        self
    }

    #[must_use]
    pub fn reduce(self) -> Shape {
        if self.rank > 1 {
            self.rremove(0)
        } else {
            self.with_cols(1)
        }
    }

    #[must_use]
    #[inline]
    pub fn insert(self, axis: usize, len: usize) -> Shape {
        let axis = self.rank - axis;

        self.rinsert(axis, len)
    }

    #[must_use]
    #[inline]
    pub fn rinsert(mut self, axis: usize, len: usize) -> Shape {
        assert!(axis + 1 < Self::MAX_RANK);

        for i in (axis..self.rank).rev() {
            self.dims[i + 1] = self.dims[i];
        }

        self.dims[axis] = len as u32;
        self.rank += 1;

        self
    }

    #[inline]
    pub(super) fn next_index(&self, index: &mut [usize]) {
        for i in 0..index.len() {
            let value = (index[i] + 1) % self.dims[i].max(1) as usize;

            index[i] = value;

            if value != 0 {
                break;
            }
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
        let mut dims = [0; Self::MAX_RANK];
        dims[0] = value as u32;

        Shape {
            dims,
            rank: 1,
        }
    }
}

impl From<&[usize]> for Shape {
    fn from(value: &[usize]) -> Self {
        let mut dims = [0; Self::MAX_RANK];

        for (i, dim) in value.iter().rev().enumerate() {
            dims[i] = *dim as u32;
        }

        Shape {
            dims,
            rank: value.len(),
        }
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(value: [usize; N]) -> Self {
        let mut dims = [0; Self::MAX_RANK];

        for (i, dim) in value.iter().rev().enumerate() {
            dims[i] = *dim as u32;
        }

        Shape {
            dims,
            rank: value.len(),
        }
    }
}

impl From<Vec<usize>> for Shape {
    fn from(value: Vec<usize>) -> Self {
        let mut dims = [0; Self::MAX_RANK];

        for (i, v) in value.iter().rev().enumerate() {
            assert!(*v > 0);
            dims[i] = *v as u32;
        }

        Shape {
            dims,
            rank: value.len(),
        }
    }
}

impl<T: Type> Tensor<T> {
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
}

impl<T: Type + Clone> Tensor<T> {
    #[inline]
    #[must_use]
    pub fn reshape(self, shape: impl Into<Shape>) -> Tensor<T> {
        let shape = shape.into();

        assert_eq!(shape.size(), self.size(), "shape size must match {:?} new={:?}", 
            self.shape().as_vec(), shape.as_vec()
        );

        Self { shape, ..self }
    }

    #[must_use]
    pub fn expand_dims(self, axis: impl Into<Axis>) -> Tensor<T> {
        let axis : Axis = axis.into();

        let axis = axis.get_axis().unwrap_or(0);

        let shape = self.shape().expand_dims(axis);

        self.reshape(shape)
    }

    #[inline]
    #[must_use]
    pub fn flatten(self) -> Tensor<T> {
        let size = self.size();

        self.reshape([size])
    }

    #[inline]
    #[must_use]
    pub fn squeeze(self) -> Tensor<T> {
        let shape = self.shape().squeeze(None);

        self.reshape(shape)
    }

    #[inline]
    #[must_use]
    pub fn squeeze_axis(self, axis: impl Into<Axis>) -> Tensor<T> {
        let shape = self.shape().squeeze(axis);

        self.reshape(shape)
    }
}

#[cfg(test)]
mod test {
    use crate::ten;

    use super::Shape;

    #[test]
    fn shape_from_slice() {
        let shape = Shape::from([]);
        assert_eq!(shape.rank(), 0);
        assert_eq!(shape.size(), 1);
        assert_eq!(shape.cols(), 0);
        assert_eq!(shape.rows(), 0);
        assert_eq!(shape.as_vec(), vec![]);

        let shape = Shape::from([4]);
        assert_eq!(shape.rank(), 1);
        assert_eq!(shape.size(), 4);
        assert_eq!(shape.cols(), 4);
        assert_eq!(shape.rows(), 0);
        assert_eq!(shape.dim(0), 4);
        assert_eq!(shape.rdim(0), 4);
        assert_eq!(shape.as_vec(), vec![4]);

        let shape = Shape::from([2, 4]);
        assert_eq!(shape.rank(), 2);
        assert_eq!(shape.size(), 8);
        assert_eq!(shape.cols(), 4);
        assert_eq!(shape.rows(), 2);
        assert_eq!(shape.dim(0), 2);
        assert_eq!(shape.dim(1), 4);
        assert_eq!(shape.rdim(0), 4);
        assert_eq!(shape.rdim(1), 2);
        assert_eq!(shape.as_vec(), vec![2, 4]);

    }
    #[test]
    fn shape_debug() {
        assert_eq!(format!("{:?}", Shape::from([])), "Shape { dims: [0, 0, 0, 0, 0, 0], rank: 0 }");
        assert_eq!(format!("{:?}", Shape::from([4])), "Shape { dims: [4, 0, 0, 0, 0, 0], rank: 1 }");
        assert_eq!(format!("{:?}", Shape::from([4, 2])), "Shape { dims: [2, 4, 0, 0, 0, 0], rank: 2 }");
    }

    #[test]
    fn shape_insert() {
        assert_eq!(Shape::from([1]).insert(0, 4).as_vec(), vec![4, 1]);
        assert_eq!(Shape::from([1]).insert(1, 4).as_vec(), vec![1, 4]);

        assert_eq!(Shape::from([1, 2]).insert(0, 4).as_vec(), vec![4, 1, 2]);
        assert_eq!(Shape::from([1, 2]).insert(1, 4).as_vec(), vec![1, 4, 2]);
        assert_eq!(Shape::from([1, 2]).insert(2, 4).as_vec(), vec![1, 2, 4]);
    }
    
    #[test]
    fn test_expand_dims() {
        assert_eq!(ten![1., 2.].expand_dims(0), ten![[1., 2.]]);
        assert_eq!(ten![1., 2.].expand_dims(1), ten![[1.], [2.]]);
        assert_eq!(ten![1., 2.].expand_dims(-1), ten![[1.], [2.]]);

        assert_eq!(
            ten![[1., 2.], [3., 4.]].expand_dims(0), 
            ten![[[1., 2.], [3., 4.]]]
        );

        assert_eq!(
            ten![[1., 2.], [3., 4.]].expand_dims(1), 
            ten![[[1., 2.]], [[3., 4.]]]
        );

        assert_eq!(
            ten![[1., 2.], [3., 4.]].expand_dims(2), 
            ten![[[1.], [2.]], [[3.], [4.]]]
        );

        assert_eq!(
            ten![[1., 2.], [3., 4.]].expand_dims(-1), 
            ten![[[1.], [2.]], [[3.], [4.]]]
        );

        assert_eq!(
            ten![[1., 2.], [3., 4.]].expand_dims(-2), 
            ten![[[1., 2.]], [[3., 4.]]]
        );
    }
}
