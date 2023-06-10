use super::{Bounds, CoordMarker};


pub struct Canvas {
    bounds: Bounds<Canvas>,
    scale_factor: f32,
}

impl Canvas {
    pub fn new(
        bounds: impl Into<Bounds<Canvas>>,
        scale_factor: f32,
    ) -> Self {
        Self {
            bounds: bounds.into(),
            scale_factor,
        }
    }

    #[inline]
    pub fn bounds(&self) -> &Bounds<Canvas> {
        &self.bounds
    }

    #[inline]
    pub fn scale_factor(&self) -> f32 {
        self.scale_factor
    }

    #[inline]
    pub fn to_px(&self, size: f32) -> f32 {
        self.scale_factor * size
    }

    pub(crate) fn set_bounds(&mut self, bounds: impl Into<Bounds<Canvas>>) {
        self.bounds = bounds.into();
    }

    pub(crate) fn set_scale_factor(&mut self, scale_factor: f32) {
        assert!(scale_factor > 0.);

        self.scale_factor = scale_factor;
    }
}

impl CoordMarker for Canvas {}
