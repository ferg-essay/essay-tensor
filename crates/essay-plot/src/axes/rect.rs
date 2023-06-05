use core::fmt;

use super::Affine2d;

#[derive(Clone)]
pub struct Rect {
    left: f32,
    bottom: f32,
    width: f32,
    height: f32,
}

impl Rect {
    pub fn new(left: f32, bottom: f32, width: f32, height: f32) -> Self {
        Self {
            left,
            bottom,
            width,
            height
        }
    }

    #[inline]
    pub fn left(&self) -> f32 {
        self.left
    }

    #[inline]
    pub fn bottom(&self) -> f32 {
        self.bottom
    }

    #[inline]
    pub fn width(&self) -> f32 {
        self.width
    }

    #[inline]
    pub fn height(&self) -> f32 {
        self.height
    }

    ///
    /// Transformation from unit square to this rectangle
    /// 
    pub fn transform_from_unit(&self) -> Affine2d {
        Affine2d::new(
            self.width, 0., self.left,
            0., self.height, self.bottom,
        )
    }

    ///
    /// Transformation from unit square to this rectangle, only
    /// for scale.
    /// 
    pub fn transform_from_unit_scale(&self) -> Affine2d {
        Affine2d::new(
            self.width, 0., 0.,
            0., self.height, 0.,
        )
    }

    ///
    /// Transformation to unit square from this rectangle
    /// 
    pub fn transform_to_unit(&self) -> Affine2d {
        let w_scale = self.width.recip();
        let h_scale = self.width.recip();

        Affine2d::new(
            w_scale, 0., - self.left * w_scale,
            0., h_scale, - self.bottom * h_scale,
        )
    }
}

impl fmt::Debug for Rect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Rect({},{};{}x{})",
            self.left(),
            self.bottom(),
            self.width(),
            self.height())
    }
}

impl From<(f32, f32, f32, f32)> for Rect {
    fn from(value: (f32, f32, f32, f32)) -> Self {
        Rect::new(value.0, value.1, value.2, value.3)
    }
}

impl From<[f32; 4]> for Rect {
    fn from(value: [f32; 4]) -> Self {
        Rect::new(value[0], value[1], value[2], value[3])
    }
}

impl From<Vec<f32>> for Rect {
    fn from(value: Vec<f32>) -> Self {
        assert!(value.len() == 4);

        Rect::new(value[0], value[1], value[2], value[3])
    }
}
