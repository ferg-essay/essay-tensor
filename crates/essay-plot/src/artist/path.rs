use std::marker::PhantomData;

use essay_tensor::Tensor;

use crate::axes::{CoordMarker, Data};

pub struct Path<M: CoordMarker = Data> {
    points: Tensor,
    codes: Vec<PathCode>,

    marker: PhantomData<M>,
}

impl<M: CoordMarker> Path<M> {
    pub fn new(points: impl Into<Tensor>, codes: Vec<PathCode>) -> Self {
        let points = points.into();

        assert_eq!(points.rank(), 2);
        assert_eq!(points.cols(), 2);
        assert_eq!(points.rows(), codes.len());

        // TODO: validate bezier

        Self {
            points,
            codes,

            marker: PhantomData
        }
    }

    #[inline]
    pub fn points(&self) -> &Tensor {
        &self.points
    }

    #[inline]
    pub fn codes(&self) -> &Vec<PathCode> {
        &self.codes
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PathCode {
    MoveTo,
    LineTo,
    BezierQuadratic,
    BezierCubic
}