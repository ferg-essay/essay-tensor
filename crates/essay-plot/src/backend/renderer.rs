use essay_tensor::Tensor;

pub trait Renderer {
    fn draw_path(
        &mut self, 
        // gc: &Gc, 
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

pub struct Path {

}

pub struct Affine2d {
    tensor: Tensor,
}