use core::fmt;
use std::marker::PhantomData;

use essay_tensor::{Tensor, tf32};

use crate::device::Device;

use super::{Rect, affine::{Point, Data, CoordMarker}, Affine2d};

///
/// Boundary box consisting of two unordered points
/// 
#[derive(Clone, PartialEq)]
pub struct Bounds<M: CoordMarker = Data> {
    p0: Point,
    p1: Point,

    marker: PhantomData<M>,
}

impl<M: CoordMarker> Bounds<M> {
    pub fn new(p0: Point, p1: Point) -> Self {
        Self {
            p0,
            p1,
            marker: PhantomData,
        }
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

    pub fn to_rect(&self) -> Rect {
        Rect::new(
            self.xmin(), 
            self.ymin(), 
            self.xmax() - self.xmin(),
            self.ymax() - self.ymin(),
        )
    }

    pub fn corners(&self) -> Tensor {
        tf32!([
            [self.p0.x(), self.p0.y()],
            [self.p0.x(), self.p1.y()],
            [self.p1.x(), self.p1.y()],
            [self.p1.x(), self.p0.y()],
        ])
    }
}

impl Bounds<Data> {
    pub fn unit() -> Self {
        Self::new(Point(0., 0.), Point(1., 1.))
    }

    pub fn to_device(&self, transform: &Affine2d) -> Bounds<Device> {
        Bounds::new(
            transform.transform_point(self.p0),
            transform.transform_point(self.p1),
        )
    }
}

impl<M: CoordMarker> fmt::Debug for Bounds<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // TODO: add marker to debug?
        f.debug_struct("BoundBox")
            .field("p0", &self.p0)
            .field("p1", &self.p1)
            .finish()
    }
}

impl From<Rect> for Bounds {
    fn from(value: Rect) -> Self {
        Bounds::from_bounds(
            value.left(), 
            value.bottom(), 
                value.width(),
                value.height()
        )
    }
}

impl From<[f32; 4]> for Bounds {
    fn from(value: [f32; 4]) -> Self {
        Bounds::new(
            Point(value[0], value[1]),
            Point(value[2], value[3]),
        )
    }
}

impl From<Tensor> for Bounds {
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

impl From<Bounds> for Tensor {
    fn from(value: Bounds) -> Self {
        value.corners()
    }
}
