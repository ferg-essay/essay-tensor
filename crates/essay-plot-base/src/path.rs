use core::fmt;
use std::{marker::PhantomData};

use essay_tensor::{prelude::*};

use crate::{Coord, Affine2d, Bounds, Point, affine};

pub struct Path<M: Coord> {
    codes: Vec<PathCode>,

    marker: PhantomData<M>,
}

impl<M: Coord> Path<M> {
    pub fn new(codes: Vec<PathCode>) -> Self {
        Self {
            codes,

            marker: PhantomData
        }
    }

    pub fn lines(points: impl Into<Tensor>) -> Self {
        let points = points.into();

        assert_eq!(points.rank(), 2);
        assert_eq!(points.cols(), 2);

        let mut codes = Vec::<PathCode>::new();

        codes.push(PathCode::MoveTo(Point(points[0], points[1])));

        let len = points.rows();

        for i in 1..len {
            codes.push(PathCode::LineTo(Point(points[(i, 0)], points[(i, 1)])));
        }

        Self::new(codes)
    }

    pub fn closed_poly(points: impl Into<Tensor>) -> Self {
        let points = points.into();

        assert_eq!(points.rank(), 2);
        assert_eq!(points.cols(), 2);

        let mut codes = Vec::<PathCode>::new();

        codes.push(PathCode::MoveTo(Point(points[0], points[1])));

        let len = points.rows() - 1;

        for i in 1..len {
            codes.push(PathCode::LineTo(Point(points[(i, 0)], points[(i, 1)])));
        }

        codes.push(PathCode::ClosePoly(Point(points[(len, 0)], points[(len, 1)])));

        assert!(codes.len() == points.rows());

        Self {
            codes,

            marker: PhantomData
        }
    }

    #[inline]
    pub fn codes(&self) -> &Vec<PathCode> {
        &self.codes
    }

    pub fn is_closed_path(&self) -> bool {
        if let PathCode::ClosePoly(_) = self.codes[self.codes.len() - 1] {
            true
        } else {
            false
        }
    }

    pub fn get_bounds(&self) -> Bounds<M> {
        let mut bounds = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];

        for code in &self.codes {
            let point = match code {
                PathCode::MoveTo(p1) => p1,
                PathCode::LineTo(p1) => p1,
                PathCode::Bezier2(_, p2) => p2,
                PathCode::Bezier3(_, _, p3) => p3,
                PathCode::ClosePoly(p1) => p1,
            };

            bounds[0] = f32::min(bounds[0], point.x());
            bounds[1] = f32::min(bounds[1], point.y());
            bounds[2] = f32::max(bounds[2], point.x());
            bounds[3] = f32::max(bounds[3], point.y());
        }

        Bounds::<M>::new(Point(bounds[0], bounds[1]), Point(bounds[2], bounds[3]))
    }

    pub fn transform<C: Coord>(&self, affine: &Affine2d) -> Path<C> {
        let mut codes = Vec::<PathCode>::new();

        for code in &self.codes {
            let code = match code {
                PathCode::MoveTo(p0) => {
                    PathCode::MoveTo(affine.transform_point(*p0))
                }
                PathCode::LineTo(p1) => {
                    PathCode::LineTo(affine.transform_point(*p1))
                }
                PathCode::Bezier2(p1, p2) => {
                    PathCode::Bezier2(
                        affine.transform_point(*p1),
                        affine.transform_point(*p2),
                    )
                }
                PathCode::Bezier3(p1, p2, p3) => {
                    PathCode::Bezier3(
                        affine.transform_point(*p1),
                        affine.transform_point(*p2),
                        affine.transform_point(*p3),
                    )
                }
                PathCode::ClosePoly(p1) => {
                    PathCode::ClosePoly(affine.transform_point(*p1))
                }
            };

            codes.push(code);
        }

        Path::new(codes)
    }

    pub fn translate<C: Coord>(&self, x: f32, y: f32) -> Path<C> {
        self.transform(&affine::translate(x, y))
    }

    pub fn scale<C: Coord>(&self, scale_x: f32, scale_y: f32) -> Path<C> {
        self.transform(&affine::scale(scale_x, scale_y))
    }

    pub fn rotate<C: Coord>(&self, theta: f32) -> Path<C> {
        self.transform(&affine::rotate(theta))
    }

    pub fn rotate_deg<C: Coord>(&self, deg: f32) -> Path<C> {
        self.transform(&affine::rotate_deg(deg))
    }
}

impl<M: Coord> Clone for Path<M> {
    fn clone(&self) -> Self {
        Self { 
            codes: self.codes.clone(), 
            marker: self.marker.clone() 
        }
    }
}

impl<M: Coord> fmt::Debug for Path<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.codes.len() {
            0 => {
                write!(f, "Path[]")
            },
            1 => {
                write!(f, "Path[{:?}]", self.codes[0])
            },
            2 => {
                write!(f, "Path[{:?}, {:?}]", self.codes[0], self.codes[1])
            },
            n => {
                write!(f, "Path[{:?}, {:?}, ..., {:?}]",
                    self.codes[0], self.codes[1], self.codes[n - 1])
            }
        }
    }
}

impl<const N: usize, M: Coord> From<[PathCode; N]> for Path<M> {
    fn from(value: [PathCode; N]) -> Self {
        let codes = Vec::<PathCode>::from(value);

        Self {
            codes,
            marker: PhantomData,
        }
    }
}

impl<M: Coord> From<&[[f32; 2]]> for Path<M> {
    fn from(value: &[[f32; 2]]) -> Self {
        let mut codes = Vec::<PathCode>::new();

        let mut is_first = true;
        for point in value {
            if is_first {
                codes.push(PathCode::MoveTo(point.into()));
                is_first = false;
            }
        }

        Self {
            codes,
            marker: PhantomData,
        }
    }
}

impl<const N: usize, M: Coord> From<[[f32; 2]; N]> for Path<M> {
    fn from(value: [[f32; 2]; N]) -> Self {
        let mut codes = Vec::<PathCode>::new();

        let mut is_first = true;
        for point in value {
            if is_first {
                codes.push(PathCode::MoveTo(point.into()));
                is_first = false;
            } else {
                codes.push(PathCode::LineTo(point.into()));
            }
        }

        Self {
            codes,
            marker: PhantomData,
        }
    }
}

impl<const N: usize, M: Coord> From<[Point; N]> for Path<M> {
    fn from(value: [Point; N]) -> Self {
        let mut codes = Vec::<PathCode>::new();

        let mut is_first = true;
        for point in value {
            if is_first {
                codes.push(PathCode::MoveTo(point.into()));
                is_first = false;
            } else {
                codes.push(PathCode::LineTo(point.into()));
            }
        }

        Self {
            codes,
            marker: PhantomData,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PathCode {
    MoveTo(Point),
    LineTo(Point),
    Bezier2(Point, Point),
    Bezier3(Point, Point, Point),
    ClosePoly(Point),
}

impl PathCode {
    pub fn tail(&self) -> Point {
        match self {
            PathCode::MoveTo(p0) => *p0,
            PathCode::LineTo(p1) => *p1,
            PathCode::Bezier2(_, p2) => *p2,
            PathCode::Bezier3(_, _, p3) => *p3,
            PathCode::ClosePoly(p1) => *p1,
        }
    }
}