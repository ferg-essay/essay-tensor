use std::{marker::PhantomData, f32::consts::{PI, TAU}};

use essay_tensor::{prelude::*, init::linspace, tensor::TensorVec};

use crate::frame::{CoordMarker, Data, Affine2d, Bounds, Point, Unit};

pub struct Path<M: CoordMarker = Data> {
    codes: Vec<PathCode>,

    marker: PhantomData<M>,
}

impl<M: CoordMarker> Path<M> {
    pub fn new(codes: Vec<PathCode>) -> Self {
        Self {
            codes,

            marker: PhantomData
        }
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

        codes.push(PathCode::ClosePoly(Point(points[(len - 1, 0)], points[(len - 1, 1)])));

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

    pub(crate) fn is_closed_path(&self) -> bool {
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

    pub fn transform<C: CoordMarker>(&self, transform: &Affine2d) -> Path<C> {
        let mut codes = Vec::<PathCode>::new();
        println!("Transform {:?}", self.codes);
        for code in &self.codes {
            let code = match code {
                PathCode::MoveTo(p0) => {
                    PathCode::MoveTo(transform.transform_point(*p0))
                },
                PathCode::LineTo(p1) => {
                    PathCode::LineTo(transform.transform_point(*p1))
                }
                PathCode::Bezier2(p1, p2) => {
                    PathCode::Bezier2(
                        transform.transform_point(*p1),
                        transform.transform_point(*p2),
                    )
                },
                PathCode::Bezier3(p1, p2, p3) => {
                    PathCode::Bezier3(
                        transform.transform_point(*p1),
                        transform.transform_point(*p2),
                        transform.transform_point(*p3),
                    )
                }
                PathCode::ClosePoly(p1) => {
                    PathCode::ClosePoly(transform.transform_point(*p1))
                }
            };

            codes.push(code);
        }
        println!("Post-Transform {:?}", codes);

        Path::new(codes)
    }
}

impl Path<Unit> {
    pub fn unit() -> Path<Unit> {
        Path::new(vec![
            PathCode::MoveTo(Point(0.0, 0.0)),
            PathCode::LineTo(Point(0.0, 1.0)),
            PathCode::LineTo(Point(1.0, 1.0)),
            PathCode::ClosePoly(Point(1.0, 0.0)),
        ])
    }

    pub fn wedge(angle: Angle) -> Path<Unit> {
        let halfpi = 0.5 * PI;

        let (t0, t1) = angle.to_radians();

        let t1 = if t0 < t1 { t1 } else { t1 + TAU };

        // TODO: 
        let n = 2.0f32.powf(((t1 - t0) / halfpi).ceil()) as usize;

        let steps = linspace(t0, t1, n + 1);

        let cos = steps.cos();
        let sin = steps.sin();

        let dt = (t1 - t0) / n as f32;
        let t = (0.5 * dt).tan();
        let alpha = dt.sin() * ((4. + 3. * t * t).sqrt() - 1.) / 3.;
        // let mut vec = TensorVec::<[f32; 2]>::new();
        let mut codes = Vec::new();

        codes.push(PathCode::MoveTo(Point(cos[0], sin[0])));

        for i in 1..=n {
            // TODO: switch to quad bezier
            // vec.push([cos[i], sin[i]]);
            codes.push(PathCode::Bezier3(
                Point(
                    cos[i - 1] - alpha * sin[i - 1], 
                    sin[i - 1] + alpha * cos[i - 1]
                ),
                Point(
                    cos[i] + alpha * sin[i], 
                    sin[i] - alpha * cos[i]
                ),
                Point(cos[i], sin[i])
            ));
        }

        // vec.push([0., 0.]);
        codes.push(PathCode::ClosePoly(Point(0., 0.)));

        //let path = Path::new(vec.into_tensor(), codes);
        let path = Path::new(codes);

        path
    }
}

impl<const N: usize, M: CoordMarker> From<[PathCode; N]> for Path<M> {
    fn from(value: [PathCode; N]) -> Self {
        let mut codes = Vec::<PathCode>::from(value);

        Self {
            codes,
            marker: PhantomData,
        }
    }
}

impl<M: CoordMarker> From<&[[f32; 2]]> for Path<M> {
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

impl<const N: usize, M: CoordMarker> From<[[f32; 2]; N]> for Path<M> {
    fn from(value: [[f32; 2]; N]) -> Self {
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

// angle in [0., 1.]
#[derive(Clone, Copy, Debug)]
pub struct Angle(pub f32, pub f32);

impl Angle {
    pub fn to_radians(&self) -> (f32, f32) {
        (unit_to_radians(self.0), unit_to_radians(self.1))
    }
}

fn unit_to_radians(unit: f32) -> f32 {
    ((1.25 - unit) % 1.) * 2. * PI
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