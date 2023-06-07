use essay_tensor::Tensor;

use crate::{artist::{Path, StyleOpt}, figure::{Affine2d, Bounds, Point, Data}};

use super::{GraphicsContext, Device};

pub trait Renderer {
    ///
    /// Returns the boundary of the canvas, usually in pixels or points.
    ///
    fn get_canvas_bounds(&self) -> Bounds<Device> {
        Bounds::new(Point(0., 0.), Point(1., 1.))
    }

    fn new_gc(&mut self) -> GraphicsContext {
        GraphicsContext::default()
    }

    fn draw_path(
        &mut self, 
        style: &dyn StyleOpt, 
        path: &Path<Data>, 
        to_device: &Affine2d,
        clip: &Bounds<Device>,
    ) -> Result<(), RenderErr>;
}

#[derive(Debug)]
pub enum RenderErr {
    NotImplemented,
}
