use essay_tensor::Tensor;

use crate::{artist::Path, figure::{Affine2d, Bounds, Point, Data}};

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
        _gc: &GraphicsContext, 
        _path: &Path<Data>, 
        _to_device: &Affine2d,
        _clip: &Bounds<Device>,
    ) -> Result<(), RenderErr> {
        Err(RenderErr::NotImplemented)
    }
}

#[derive(Debug)]
pub enum RenderErr {
    NotImplemented,
}
