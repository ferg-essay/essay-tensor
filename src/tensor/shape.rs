use std::{cmp, ops::Index, slice::SliceIndex};

#[derive(Debug, Clone, PartialEq)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    pub fn scalar() -> Self {
        Self {
            dims: Vec::new(),
        }
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.dims.iter().product::<usize>() as usize
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    #[inline]
    pub fn dim(&self, i: usize) -> usize {
        self.dims[i] as usize
    }

    #[inline]
    pub fn dim_rev(&self, i: usize) -> usize {
        self.dims[self.dims.len() - 1 - i] as usize
    }

    #[inline]
    pub fn idim(&self, i: isize) -> usize {
        let i = (i + self.rank() as isize) as usize % self.rank();
        
        self.dims[i] as usize
    }

    #[inline]
    pub fn dim_tail(&self) -> usize {
        let rank = self.rank();

        if rank > 0 {
            self.dims[rank - 1] as usize
        } else {
            1
        }
    }

    #[inline]
    pub fn cols(&self) -> usize {
        let rank = self.rank();

        if rank > 0 {
            self.dims[rank - 1] as usize
        } else {
            1
        }
    }
    
    #[inline]
    pub fn with_col(&self, col: usize) -> Self {
        let rank = self.rank();

        assert!(rank > 0);

        let mut dims = self.dims.clone();
        dims[rank - 1] = col;

        Self {
            dims,
        }
    }

    #[inline]
    pub fn rows(&self) -> usize {
        let rank = self.rank();

        if rank > 1 {
            self.dims[rank - 2] as usize
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

    pub fn check_broadcast(&self, b: &Shape) {
        let min_rank = cmp::min(self.rank(), b.rank());
        for i in 0..min_rank {
            assert_eq!(
                self.dim_rev(i), b.dim_rev(i), 
                "broadcast ranks must match a={:?} b={:?}",
                self.as_slice(), b.as_slice(),
            );
        }
    }

    pub fn broadcast_to(&self, b: &Shape) -> Self {
        let min_rank = cmp::min(self.rank(), b.rank());
        for i in 0..min_rank {
            assert_eq!(
                self.dim_rev(i), b.dim_rev(i), 
                "broadcast ranks must match a={:?} b={:?}",
                self.as_slice(), b.as_slice(),
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

    #[inline]
    pub fn expand_dims(&self, axis: isize) -> Self {
        let mut vec = Vec::from(self.as_slice());

        let index = if axis >= 0  { axis } else { vec.len() as isize + axis + 1 } as usize;
        assert!(index <= vec.len(), "expand_dims axis is invalid {} in {:?}", axis, self.as_slice());

        vec.insert(index, 1);

        Self::from(vec)
    }

    pub fn squeeze(&self, axis: &Option<isize>) -> Self {
        let mut vec = Vec::<usize>::new();
        match axis {
            None => {
                for dim in self.as_slice() {
                    if *dim != 1 {
                        vec.push(*dim)
                    }
                }
            },
            Some(axis) => {
                let len = self.as_slice().len();
                let axis = (axis + len as isize) % len as isize;
                let axis = axis as usize;
                
                let mut vec = Vec::<usize>::new();
                for (i, dim) in self.as_slice().iter().enumerate() {
                    if i != axis || *dim != 1 {
                        vec.push(*dim)
                    }
                }
            }
        };

        Self::from(vec)
    }

    pub fn pop(&self) -> Shape {
        let mut vec = self.dims.clone();
        vec.pop();
        Self::from(vec)
    }

    pub fn reduce(&self) -> Shape {
        let mut vec = self.dims.clone();
        vec.pop();

        if vec.len() == 0 {
            vec.push(1);
        }
        
        Self::from(vec)
    }

    pub fn remove(&self, axis: usize) -> Shape {
        let mut vec = self.dims.clone();
        vec.remove(axis);
        Self::from(vec)
    }

    pub fn insert(&self, axis: usize, len: usize) -> Shape {
        let mut vec = self.dims.clone();
        vec.insert(axis, len);
        Self::from(vec)
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
