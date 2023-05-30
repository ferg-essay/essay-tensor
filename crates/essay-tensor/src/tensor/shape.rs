use std::{slice::SliceIndex, cmp, ops::Index};


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

/*
impl<I> Index<I> for Shape 
where
    I:SliceIndex<[usize], Output=Vec<usize>>
{
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.dim(index)
    }
}
*/
