use essay_tensor::{prelude::*, tensor::TensorUninit};

pub struct Affine2d {
    mat: Tensor,
}

impl Affine2d {
    pub fn new(
        a: f32, b: f32, c: f32, 
        d: f32, e: f32, f: f32
    ) -> Affine2d {
        let mat = tf32!([
            [a, b, c],
            [d, e, f],
            [0., 0., 1.],
        ]); 

        Self {
            mat
        }
    }

    pub fn eye() -> Self {
        // TODO: use Tensor::eye
        let mat = tf32!([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ]); 

        Self {
            mat
        }
    }

    pub fn translate(&self, x: f32, y: f32) -> Self {
        let translate = tf32!([
            [1., 0., x],
            [0., 1., y],
            [0., 0., 1.],
        ]); 

        Self {
            mat: compose(&translate, &self.mat),
        }
    }

    pub fn scale(&self, sx: f32, sy: f32) -> Self {
        let scale = tf32!([
            [sx, 0., 0.],
            [0., sy, 0.],
            [0., 0., 1.],
        ]); 

        Self {
            mat: compose(&scale, &self.mat),
        }
    }

    pub fn rotate(&self, theta: f32) -> Self {
        let sin = theta.sin();
        let cos = theta.cos();

        let rot = tf32!([
            [cos, -sin, 0.],
            [sin, cos, 0.],
            [0., 0., 1.],
        ]); 

        Self {
            mat: compose(&rot, &self.mat),
        }
    }

    pub fn rotate_around(&self, x: f32, y: f32, theta: f32) -> Self {
        self.translate(-x, -y).rotate(theta).translate(x, y)
    }

    pub fn rotate_deg(&self, deg: f32) -> Self {
        self.rotate(deg.to_radians())
    }

    pub fn rotate_unit(&self, unit: f32) -> Self {
        self.rotate((0.25 - unit) * std::f32::consts::PI)
    }

    pub fn matmul(&self, y: &Affine2d) -> Self {
        Self {
            mat: compose(&self.mat, &y.mat),
        }
    }

    pub fn transform(&self, points: Tensor) -> Tensor {
        assert!(points.rank() == 2);
        assert!(points.cols() == 2);

        let n = points.rows();

        unsafe {
            let mut out = TensorUninit::<f32>::new(2 * n);

            let mat = self.mat.as_slice();
            let xy = points.as_slice();
            let o = out.as_mut_slice();

            for i in 0..n {
                let x = xy[2 * i];
                let y = xy[2 * i + 1];

                let x1 = x * mat[0] + y * mat[1] + mat[2];
                let y1 = x * mat[3] + y * mat[4] + mat[5];

                o[2 * i] = x1;
                o[2 * i + 1] = y1;
            }

            Tensor::from_uninit(out, points.shape())
        }
    }

    pub fn transform_point(&self, point: Point) -> Point {
        let mat = self.mat.as_slice();

        let Point(x, y) = point;

        Point(
            x * mat[0] + y * mat[1] + mat[2],
            x * mat[3] + y * mat[4] + mat[5],
        )
    }
}

fn compose(x: &Tensor, y: &Tensor) -> Tensor {
    assert_eq!(x.shape().as_slice(), &[3, 3]);
    assert_eq!(y.shape().as_slice(), &[3, 3]);

    unsafe {
        let mut out = TensorUninit::<f32>::new(9);

        let mut o = out.as_mut_slice();
        let x = x.as_slice();
        let y = y.as_slice();

        o[0] = x[0] * y[0] + x[1] * y[3];
        o[1] = x[0] * y[1] + x[1] * y[4];
        o[2] = x[0] * y[2] + x[1] * y[5] + x[2];

        o[3] = x[3] * y[0] + x[4] * y[3];
        o[4] = x[3] * y[1] + x[4] * y[4];
        o[5] = x[3] * y[2] + x[4] * y[5] + x[5];

        o[6] = 0.;
        o[7] = 0.;
        o[8] = 1.;

        Tensor::from_uninit(out, [3, 3])
    }
}

pub trait CoordMarker {}

#[derive(Clone, Copy, Debug)]
pub struct Data {}

impl CoordMarker for Data {
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point(pub f32, pub f32);

impl Point {
    #[inline]
    pub fn x(&self) -> f32 {
        self.0
    }

    #[inline]
    pub fn y(&self) -> f32 {
        self.1
    }
}
