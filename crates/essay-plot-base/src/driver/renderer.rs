use essay_tensor::Tensor;

use crate::{Path, StyleOpt, TextStyle, Affine2d, Bounds, Point, Canvas};

pub trait Renderer {
    ///
    /// Returns the boundary of the canvas, usually in pixels or points.
    ///
    fn get_canvas_bounds(&self) -> Bounds<Canvas> {
        Bounds::unit()
    }

    fn to_px(&self, size: f32) -> f32 {
        size
    }

    fn draw_path(
        &mut self, 
        style: &dyn StyleOpt, 
        path: &Path<Canvas>, 
        to_canvas: &Affine2d,
        clip: &Bounds<Canvas>,
    ) -> Result<(), RenderErr>;

    fn draw_markers(
        &mut self, 
        marker: &Path<Canvas>, 
        xy: &Tensor,
        style: &dyn StyleOpt, 
        clip: &Bounds<Canvas>,
    ) -> Result<(), RenderErr>;

    fn draw_text(
        &mut self, 
        xy: Point, // location in Canvas coordinates
        text: &str,
        angle: f32,
        style: &dyn StyleOpt, 
        text_style: &TextStyle,
        clip: &Bounds<Canvas>,
    ) -> Result<(), RenderErr>;

    fn draw_triangles(
        &mut self,
        vertices: Tensor<f32>,  // Nx2 x,y in canvas coordinates
        colors: Tensor<u32>,    // N in rgba
        triangles: Tensor<u32>, // Mx3 vertex indices
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
