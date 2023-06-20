use essay_tensor::Tensor;

use crate::{Path, PathOpt, TextStyle, Affine2d, Bounds, Point, Canvas, Clip};

pub trait Renderer {
    ///
    /// Returns the boundary of the canvas, usually in pixels or points.
    ///
    fn get_canvas(&self) -> &Canvas;

    fn to_px(&self, size: f32) -> f32 {
        size
    }

    fn draw_path(
        &mut self, 
        style: &dyn PathOpt, 
        path: &Path<Canvas>, 
        to_canvas: &Affine2d,
        clip: &Clip,
    ) -> Result<(), RenderErr>;

    fn draw_markers(
        &mut self, 
        marker: &Path<Canvas>, 
        xy: &Tensor,
        style: &dyn PathOpt, 
        clip: &Clip,
    ) -> Result<(), RenderErr>;

    fn draw_text(
        &mut self, 
        xy: Point, // location in Canvas coordinates
        text: &str,
        angle: f32,
        style: &dyn PathOpt, 
        text_style: &TextStyle,
        clip: &Clip,
    ) -> Result<(), RenderErr>;

    fn draw_triangles(
        &mut self,
        vertices: Tensor<f32>,  // Nx2 x,y in canvas coordinates
        colors: Tensor<u32>,    // N in rgba
        triangles: Tensor<u32>, // Mx3 vertex indices
        clip: &Clip,
    ) -> Result<(), RenderErr>;

    fn request_redraw(
        &mut self,
        bounds: &Bounds<Canvas>
    );
}

#[derive(Debug)]
pub enum RenderErr {
    NotImplemented,
}
