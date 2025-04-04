use std::cmp;

use super::Axis;

#[derive(Debug, Clone, PartialEq)]
pub struct Shape {
    dims: [u32; Shape::MAX_RANK],
    rank: usize,
}

impl Shape {
    pub const MAX_RANK: usize = 6;

    pub fn scalar() -> Self {
        let mut dims = [0; Self::MAX_RANK];
        dims[0] = 1;

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
    
    #[inline]
    pub fn with_rank(self, rank: usize) -> Self {
        let mut own = self;

        for i in rank..Self::MAX_RANK {
            own.dims[i] = 0;
        }

        own.rank = rank;

        own
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
    pub fn with_dim(self, i: usize, value: usize) -> Self {
        let mut dims = self.dims.clone();

        dims[self.rank - 1 - i] = value as u32;

        Self {
            dims,
            rank: self.rank
        }
    }

    ///
    /// Dimension indexed in reverse order, so 0 is cols, 1 is rows
    /// 
    #[inline]
    pub fn rdim(&self, i: usize) -> usize {
        self.dims[i] as usize
    }

    ///
    /// Dimension indexed in bottom-up, reverse order, where 0 is cols, 1 is rows
    /// 
    #[inline]
    pub fn with_rdim(self, i: usize, value: usize) -> Self {
        let mut dims = self.dims.clone();

        dims[i] = value as u32;

        Self {
            dims,
            rank: self.rank
        }
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
    pub fn with_col(self, col: usize) -> Self {
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
    pub fn with_row(&self, row: usize) -> Self {
        let mut dims = self.dims.clone();
        dims[1] = row as u32;

        Self {
            dims,
            rank: self.rank,
        }
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
        let mut size = 1;

        for i in start..end {
            size *= self.dims[i];
        }

        size as usize
    }

    #[inline]
    pub fn expand_dims(&self, axis: isize) -> Self {
        let mut vec = self.as_vec();

        let index = if axis >= 0  { axis } else { vec.len() as isize + axis + 1 } as usize;
        assert!(index <= vec.len(), "expand_dims axis is invalid {} in {:?}", axis, self.as_vec());

        vec.insert(index, 1);

        Self::from(vec)
    }

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

    pub fn reduce(&self) -> Shape {
        let mut dims = [0; Self::MAX_RANK];

        dims[0] = 1;
        for i in 0..self.rank - 1 {
            dims[i] = self.dims[i + 1];
        }

        Self {
            dims,
            rank: self.rank.max(2) - 1,
        }
    }

    pub fn remove(&self, axis: usize) -> Shape {
        let axis = self.rank - 1 - axis;
        let mut dims = [0; Self::MAX_RANK];

        dims[0] = 0;
        for i in 0..axis {
            dims[i] = self.dims[i];
        }

        for i in axis + 1..self.rank {
            dims[i - 1] = self.dims[i];
        }

        Self {
            dims,
            rank: self.rank.max(1) - 1,
        }
    }

    #[must_use]
    pub fn insert(self, axis: usize, len: usize) -> Shape {
        let axis = self.rank - axis;
        let mut dims = [0; Self::MAX_RANK];

        for i in 0..axis {
            dims[i] = self.dims[i];
        }

        dims[axis] = len as u32;

        for i in axis..self.rank {
            dims[i + 1] = self.dims[i];
        }

        Self {
            dims,
            rank: self.rank + 1,
        }
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

        for (i, v) in value.iter().rev().enumerate() {
            dims[i] = *v as u32;
        }

        Shape {
            dims,
            rank: value.len().max(1),
        }
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(value: [usize; N]) -> Self {
        let mut dims = [0; Self::MAX_RANK];

        for (i, v) in value.iter().rev().enumerate() {
            dims[i] = *v as u32;
        }

        Shape {
            dims,
            rank: value.len().max(1),
        }
    }
}

impl From<Vec<usize>> for Shape {
    fn from(value: Vec<usize>) -> Self {
        let mut dims = [0; Self::MAX_RANK];

        for (i, v) in value.iter().rev().enumerate() {
            dims[i] = *v as u32;
        }

        Shape {
            dims,
            rank: value.len().max(1),
        }
    }
}

#[cfg(test)]
mod test {
    use super::Shape;

    #[test]
    fn shape_from_slice() {
        let shape = Shape::from([]);
        assert_eq!(shape.rank(), 1);
        assert_eq!(shape.size(), 0);
        assert_eq!(shape.cols(), 0);
        assert_eq!(shape.rows(), 0);
        assert_eq!(shape.as_vec(), vec![0]);

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
        assert_eq!(format!("{:?}", Shape::from([])), "Shape { dims: [0, 0, 0, 0, 0, 0], rank: 1 }");
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
}
