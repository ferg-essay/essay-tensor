use essay_tensor::Tensor;

use crate::{artist::Path, axes::{Affine2d, Bounds, Point}};

use super::{GraphicsContext, Device};

pub trait Renderer {
    ///
    /// Returns the boundary of the canvas, usually in pixels or points.
    ///
    fn get_canvas_bounds(&self) -> Bounds<Device> {
        Bounds::new(Point(0., 0.), Point(1., 1.))
    }

    fn new_gc(&self) -> &GraphicsContext {
        todo!()
    }

    fn draw_path(
        &mut self, 
        _gc: &GraphicsContext, 
        _path: &Path<Device>, 
        _transform: &Affine2d,
    ) -> Result<(), RenderErr> {
        Err(RenderErr::NotImplemented)
    }
}

#[derive(Debug)]
pub enum RenderErr {
    NotImplemented,
}
