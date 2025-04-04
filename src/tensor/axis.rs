use crate::prelude::Shape;



#[derive(Default)]
pub struct Axis {
    axis: Option<isize>,
}

impl Axis {
    pub fn axis(axis: isize) -> Self {
        Self { axis: Some(axis) }
    }

    pub fn axis_opt(axis: Option<isize>) -> Self {
        Self { axis }
    }

    pub fn get_axis(&self) -> Option<isize> {
        self.axis
    }

    pub fn axis_from_rank(&self, rank: usize) -> usize {
        match self.axis {
            Some(axis) => (axis + rank as isize) as usize % rank,
            None => 0,
        }
    }
    
    pub(crate) fn _axis_with_shape(&self, shape: &Shape) -> usize {
        match self.axis {
            Some(axis) => {
                (axis + shape.rank() as isize) as usize % shape.rank()
            },
            None => 0
        }
    }

    pub fn reduce(&self, shape: &Shape) -> (Shape, usize, usize, usize) {
        match self.axis {
            None => (Shape::scalar(), 1, shape.size(), 1),
            Some(axis) => {
                let rank = shape.rank();
                let axis = ((axis + rank as isize) % rank as isize) as usize;
                assert!(axis < rank);

                if rank == 1 {
                    return (Shape::scalar(), 1, shape.size(), 1)
                }

                let mut vec = Vec::<usize>::new();

                let mut outer = 1;
                for i in 0..axis {
                    let dim = shape.dim(i);
                    vec.push(dim);
                    outer *= dim;
                }
                
                let mut inner = 1;
                for i in axis + 1..rank {
                    let dim = shape.dim(i);
                    vec.push(dim);
                    inner *= dim;
                }

                (Shape::from(vec), outer, shape.dim(axis), inner)
            }
        }

    }

}

pub fn axis_from_rank(axis: &Option<isize>, rank: usize) -> usize {
    match axis {
        Some(axis) => (axis + rank as isize) as usize % rank,
        None => 0,
    }
}

impl From<Option<isize>> for Axis {
    fn from(value: Option<isize>) -> Self {
        Axis::axis_opt(value)
    }
}

impl From<isize> for Axis {
    fn from(axis: isize) -> Self {
        Axis::axis(axis)
    }
}
