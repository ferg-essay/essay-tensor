use essay_tensor::Tensor;

use crate::{artist::{Path, StyleOpt}, frame::{Affine2d, Bounds, Point, Data}};

use super::{GraphicsContext, Canvas};

pub trait Renderer {
    ///
    /// Returns the boundary of the canvas, usually in pixels or points.
    ///
    fn get_canvas_bounds(&self) -> Bounds<Canvas> {
        Bounds::new(Point(0., 0.), Point(1., 1.))
    }

    fn new_gc(&mut self) -> GraphicsContext {
        GraphicsContext::default()
    }

    fn draw_path(
        &mut self, 
        style: &dyn StyleOpt, 
        path: &Path<Data>, 
        to_canvas: &Affine2d,
        clip: &Bounds<Canvas>,
    ) -> Result<(), RenderErr>;

    fn draw_text(
        &mut self, 
        style: &dyn StyleOpt, 
        xy: Point, // location in Canvas coordinates
        s: &str,
        // prop, - font properties
        // affine: &Affine2d,
        angle: f32, // rotation
        clip: &Bounds<Canvas>,
    ) -> Result<(), RenderErr> {
        println!("Draw Test {}", s);
        Ok(())
    }

    // draw_markers(path, marker_path, marker_trans)
    // i.e. markers only have location differences

    // draw_path_collection(gc, master_transform, paths, all_transforms,
    // offsets, offset_trans, facecolors, edgecolors, linewidths, linestyles

    // draw_quad_mesh(mesh_width, mesh_height, coord, offses, facecolors, edgecolors))
}

#[derive(Debug)]
pub enum RenderErr {
    NotImplemented,
}
