use std::marker::PhantomData;

use essay_tensor::Tensor;

use crate::axes::{CoordMarker, Data};

pub struct Path<M: CoordMarker = Data> {
    points: Tensor,
    codes: Vec<PathCode>,

    marker: PhantomData<M>,
}

impl Path {
    pub fn new(points: impl Into<Tensor>, codes: Vec<PathCode>) -> Path {
        let points = points.into();

        assert_eq!(points.rank(), 2);
        assert_eq!(points.cols(), 2);
        assert_eq!(points.rows(), codes.len());

        // TODO: validate bezier

        Path {
            points,
            codes,

            marker: PhantomData
        }
    }
}

pub enum PathCode {
    MoveTo,
    LineTo,
    BezierQuadratic,
    BezierCubic
}