use std::{marker::PhantomData, f32::consts::{PI, TAU}};

use essay_tensor::{prelude::*, init::linspace, tensor::TensorVec};

use crate::figure::{CoordMarker, Data, Affine2d, Bounds, Point};

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

    pub fn get_bounds(&self) -> Bounds<M> {
        let mut bounds = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];

        for point in self.points.iter_slice() {
            bounds[0] = f32::min(bounds[0], point[0]);
            bounds[1] = f32::min(bounds[1], point[1]);
            bounds[2] = f32::max(bounds[2], point[0]);
            bounds[3] = f32::max(bounds[3], point[1]);
        }

        Bounds::<M>::new(Point(bounds[0], bounds[1]), Point(bounds[2], bounds[3]))
    }

    pub fn unit<M1: CoordMarker>() -> Path<M1> {
        let points = tf32!([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0]
        ]);

        let codes = vec![
            PathCode::MoveTo,
            PathCode::LineTo,
            PathCode::ClosePoly,
        ];

        Path::new(points, codes)
    }

    pub fn wedge(angle: Angle) -> Path {
        let halfpi = 0.5 * PI;

        let (t0, t1) = angle.to_radians();
        println!("Angle {:?} Rads-0 ({:?}, {:?})", angle, t0, t1);

        let t1 = if t0 < t1 { t1 } else { t1 + TAU };

        // TODO: 
        let n = 2.0f32.powf(((t1 - t0) / halfpi).ceil()) as usize;

        let steps = linspace(t0, t1, n + 1);

        let cos = steps.cos();
        let sin = steps.sin();

        let mut vec = TensorVec::<[f32; 2]>::new();
        let mut codes = Vec::new();

        vec.push([cos[0], sin[0]]);
        codes.push(PathCode::MoveTo);

        for i in 1..=n {
            // TODO: switch to quad bezier
            vec.push([cos[i], sin[i]]);
            codes.push(PathCode::LineTo);
        }

        vec.push([0., 0.]);
        codes.push(PathCode::ClosePoly);

        let path = Path::new(vec.into_tensor(), codes);

        path
    }

    pub fn transform<C: CoordMarker>(&self, transform: &Affine2d) -> Path<C> {
        let points = transform.transform(&self.points);

        Path::new(points, self.codes.clone())
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
    MoveTo,
    LineTo,
    BezierQuadratic,
    BezierCubic,
    ClosePoly,
}