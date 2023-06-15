use core::fmt;
use std::{marker::PhantomData, any::type_name};

use essay_tensor::{Tensor, tf32};

use super::{Point, Coord, Affine2d};

///
/// Boundary box consisting of two unordered points
/// 
#[derive(PartialEq)]
pub struct Bounds<M: Coord> {
    p0: Point,
    p1: Point,

    marker: PhantomData<M>,
}

impl<M: Coord> Bounds<M> {
    pub fn new(p0: Point, p1: Point) -> Self {
        Self {
            p0,
            p1,
            marker: PhantomData,
        }
    }

    #[inline]
    pub fn extent(width: f32, height: f32) -> Self {
        Self::new(Point(0., 0.), Point(width, height))
    }

    pub fn from_bounds(
        x0: f32, 
        y0: f32, 
        width: f32, 
        height: f32
    ) -> Bounds<M> {
        Bounds {
            p0: Point(x0, y0),
            p1: Point(x0 + width, x0 + height),
            marker: PhantomData,
        }
    }

    pub fn none() -> Bounds<M> {
        Bounds {
            p0: Point(f32::MIN, f32::MIN),
            p1: Point(f32::MAX, f32::MAX),
            marker: PhantomData,
        }
    }

    pub fn zero() -> Bounds<M> {
        Bounds {
            p0: Point(0., 0.),
            p1: Point(0., 0.),
            marker: PhantomData,
        }
    }

    pub fn unit() -> Self {
        Self::new(Point(0., 0.), Point(1., 1.))
    }

    #[inline]
    pub fn is_none(&self) -> bool {
        self.p0 == Point(f32::MIN, f32::MIN) &&
            self.p1 == Point(f32::MAX, f32::MAX)
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.p0 == Point(0., 0.) && self.p1 == Point(0., 0.)
    }

    #[inline]
    pub fn x0(&self) -> f32 {
        self.p0.x()
    }

    #[inline]
    pub fn y0(&self) -> f32 {
        self.p0.y()
    }

    #[inline]
    pub fn x1(&self) -> f32 {
        self.p1.x()
    }

    #[inline]
    pub fn y1(&self) -> f32 {
        self.p1.y()
    }

    #[inline]
    pub fn xmin(&self) -> f32 {
        self.p0.x().min(self.p1.x())
    }

    #[inline]
    pub fn ymin(&self) -> f32 {
        self.p0.y().min(self.p1.y())
    }

    #[inline]
    pub fn xmax(&self) -> f32 {
        self.p0.x().max(self.p1.x())
    }

    #[inline]
    pub fn ymax(&self) -> f32 {
        self.p0.y().max(self.p1.y())
    }

    #[inline]
    pub fn xmid(&self) -> f32 {
        0.5 * (self.p0.x() + self.p1.x())
    }

    #[inline]
    pub fn ymid(&self) -> f32 {
        0.5 * (self.p0.y() + self.p1.y())
    }

    #[inline]
    pub fn width(&self) -> f32 {
        self.xmax() - self.xmin()
    }

    #[inline]
    pub fn height(&self) -> f32 {
        self.ymax() - self.ymin()
    }

    #[inline]
    pub fn contains(&self, point: Point) -> bool {
        self.contains_x(point.x()) && self.contains_y(point.y())
    }

    #[inline]
    pub fn contains_x(&self, x: f32) -> bool {
        self.x0() <= x && x <= self.x1()
            || self.x1() <= x && x <= self.x0()

    }

    #[inline]
    pub fn contains_y(&self, y: f32) -> bool {
        self.y0() <= y && y <= self.y1()
            || self.y1() <= y && y <= self.y0()

    }

    pub fn corners(&self) -> Tensor {
        tf32!([
            [self.p0.x(), self.p0.y()],
            [self.p0.x(), self.p1.y()],
            [self.p1.x(), self.p1.y()],
            [self.p1.x(), self.p0.y()],
        ])
    }

    pub fn affine_to<N>(&self, box_to: &Bounds<N>) -> Affine2d
    where
        N: Coord
    {
        let a_x0 = self.xmin();
        let a_y0 = self.ymin();

        let epsilon = f32::EPSILON;
        let a_width = self.width().max(epsilon);
        let a_height = self.height().max(epsilon);

        let b_x0 = box_to.xmin();
        let b_y0 = box_to.ymin();

        let b_width = box_to.width();
        let b_height = box_to.height();

        Affine2d::eye()
            .translate(- a_x0, - a_y0)
            .scale(b_width / a_width, b_height / a_height)
            .translate(b_x0, b_y0)
    }

    pub fn union(&self, b: &Bounds<M>) -> Self {
        Self {
            p0: Point(
                self.xmin().min(b.xmin()),
                self.ymin().min(b.ymin()),
            ),
            p1: Point(
                self.xmax().max(b.xmax()),
                self.ymax().max(b.ymax()),
            ),
            marker: PhantomData,
        }
    }
}

impl<M: Coord> Clone for Bounds<M> {
    fn clone(&self) -> Self {
        Self { 
            p0: self.p0.clone(), 
            p1: self.p1.clone(), 
            marker: self.marker.clone() 
        }
    }
}

impl<M: Coord> fmt::Debug for Bounds<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // TODO: add marker to debug?
        let name = type_name::<M>();
        let tail = name.split("::").last();
        let tail = match tail {
            Some(tail) => tail,
            None => name,
        };
        
        write!(f, 
            "Bounds<{}>({},{}; {}x{})", 
            tail,
            self.xmin(),
            self.ymin(),
            self.width(),
            self.height()
        )
    }
}

impl<M: Coord> From<()> for Bounds<M> {
    fn from(_: ()) -> Self {
        Bounds::zero()
    }
}

impl<M: Coord> From<(f32, f32)> for Bounds<M> {
    fn from(value: (f32, f32)) -> Self {
        Bounds::new(
            Point(value.0, value.1),
            Point(value.0, value.1),
        )
    }
}

impl<M: Coord> From<Point> for Bounds<M> {
    fn from(value: Point) -> Self {
        Bounds::new(
            value,
            value,
        )
    }
}

impl<M: Coord> From<[f32; 2]> for Bounds<M> {
    fn from(value: [f32; 2]) -> Self {
        Bounds::new(
            Point(0., 0.),
            Point(value[0], value[1]),
        )
    }
}

impl<M: Coord> From<[f32; 4]> for Bounds<M> {
    fn from(value: [f32; 4]) -> Self {
        Bounds::new(
            Point(value[0], value[1]),
            Point(value[2], value[3]),
        )
    }
}

impl<M: Coord> From<Tensor> for Bounds<M> {
    fn from(value: Tensor) -> Self {
        assert!(value.rank() == 2);
        assert!(value.cols() == 2);

        let mut x0 = f32::MAX;
        let mut y0 = f32::MAX;

        let mut x1 = f32::MIN;
        let mut y1 = f32::MIN;

        for point in value.iter_slice() {
            x0 = x0.min(point[0]);
            y0 = y0.min(point[1]);

            x1 = x1.max(point[0]);
            y1 = y1.max(point[1]);
        }

        Bounds {
            p0: Point(x0, y0),
            p1: Point(x1, y1),
            marker: PhantomData,
        }
    }
}

impl<M: Coord> From<Bounds<M>> for Tensor {
    fn from(value: Bounds<M>) -> Self {
        value.corners()
    }
}
