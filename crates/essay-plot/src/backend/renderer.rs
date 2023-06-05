use essay_tensor::Tensor;

use crate::{artist::Path, axes::Affine2d};

use super::GraphicsContext;

pub trait Renderer {
    fn draw_path(
        &mut self, 
        // gc: &dyn GraphicsContext, 
        path: &Path, 
        transform: &Affine2d,
    ) -> Result<(), RenderErr> {
        Err(RenderErr::NotImplemented)
    }
}

#[derive(Debug)]
pub enum RenderErr {
    NotImplemented,
}
