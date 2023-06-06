use std::marker::PhantomData;

use essay_tensor::Tensor;

use crate::figure::{CoordMarker, Data};

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

    pub fn closed_poly(points: impl Into<Tensor>) -> Self {
        let points = points.into();

        assert_eq!(points.rank(), 2);
        assert_eq!(points.cols(), 2);

        let mut codes = Vec::<PathCode>::new();
        codes.push(PathCode::MoveTo);
        for _ in 1..points.rows() - 1 {
            codes.push(PathCode::LineTo);
        }
        codes.push(PathCode::ClosePoly);

        assert!(codes.len() == points.rows());

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

    pub(crate) fn is_closed_path(&self) -> bool {
        self.codes[self.codes.len() - 1] == PathCode::ClosePoly
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PathCode {
    MoveTo,
    LineTo,
    BezierQuadratic,
    BezierCubic,
    ClosePoly,
}