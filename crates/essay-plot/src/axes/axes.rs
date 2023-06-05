use core::fmt;

use crate::{device::Renderer, figure::Figure};

use super::{rect::Rect, Bounds};

pub struct Axes {
    position: Bounds<Figure>, // position of the Axes in figure coordinates [0, 1]x[0, 1]Rect(0,0; 1x1)

    data_lim: Rect, // rectangle in data coordinates
    view_lim: Rect, // rectangle in figure coordinates [0., 1.] x [0., 1.]
}

impl Axes {
    pub fn new(bounds: impl Into<Bounds<Figure>>) -> Self {
        Self {
            position: bounds.into(),
            view_lim: [0., 0., 1., 1.].into(),
            data_lim: [-1., -1., 1., 1.].into(),
        }
    }

    /*
    /// only includes data extent, not labels or axes
    pub fn get_window_extent(&self, _renderer: Option<&dyn Renderer>) -> &Bounds {
        &self.position
    }
    */
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