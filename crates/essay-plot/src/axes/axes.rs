use core::fmt;

use crate::backend::Renderer;

use super::{rect::Rect, BoundBox};

pub struct Axes {
    bounds: BoundBox, // rectange in figure coordinates Rect(0,0; 1x1)

    data_lim: Rect, // rectange in data coordinates
    view_lim: Rect, // rectangle in figure coordinates [0., 1.] x [0., 1.]
}

impl Axes {
    pub fn new(bounds: impl Into<BoundBox>) -> Self {
        Self {
            bounds: bounds.into(),
            view_lim: [0., 0., 1., 1.].into(),
            data_lim: [-1., -1., 1., 1.].into(),
        }
    }

    /// only includes data extent, not labels or axes
    pub fn get_window_extent(&self, _renderer: Option<&dyn Renderer>) -> &BoundBox {
        &self.bounds
    }
}

impl fmt::Debug for Axes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Axes({},{},{}x{})",
            self.view_lim.left(),
            self.view_lim.bottom(),
            self.view_lim.width(),
            self.view_lim.height())
    }
}