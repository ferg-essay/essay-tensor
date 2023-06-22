use essay_plot_base::{Bounds, Point, Canvas, Clip, PathOpt, Path, Color};
use essay_tensor::{tf32, tensor::{self, TensorVec}, Tensor};

use crate::frame::Data;

use super::{Artist, PathStyle};

pub struct PColor {
}

impl Artist<Data> for PColor {
    fn update(&mut self, _canvas: &Canvas) {
    }
    
    fn get_extent(&mut self) -> Bounds<Data> {
        Bounds::new(Point(0.0, 0.0), Point(5., 5.))
    }

    fn draw(
        &mut self, 
        renderer: &mut dyn essay_plot_base::driver::Renderer,
        to_canvas: &essay_plot_base::Affine2d,
        clip: &Clip,
        style: &dyn PathOpt,
    ) {
        let vertices = tf32!([
            [0.0, 0.0], [1.0, 0.0], [0.5, 1.0],
            [1.5, 1.0]
            ]);
        let colors: Tensor<u32> = tensor!([0xff0000ff, 0xffff00ff, 0xff0000ff, 0x0080ffff]);
        let triangles: Tensor<u32> = tensor!([[0, 1, 2], [1, 3, 2]]);

        let vertices = to_canvas.transform(&vertices);

        renderer.draw_triangles(vertices, colors, triangles, clip).unwrap();
    }
}