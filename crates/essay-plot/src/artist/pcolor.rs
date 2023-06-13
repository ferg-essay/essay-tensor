use essay_plot_base::{Bounds, Point};
use essay_tensor::{tf32, tensor, Tensor};

use crate::graph::Data;

use super::ArtistTrait;

pub struct PColor {
}

impl ArtistTrait<Data> for PColor {
    fn get_bounds(&mut self) -> Bounds<Data> {
        Bounds::new(Point(0.0, 0.0), Point(1.5, 1.0))
    }

    fn draw(
        &mut self, 
        renderer: &mut dyn essay_plot_base::driver::Renderer,
        to_canvas: &essay_plot_base::Affine2d,
        clip: &essay_plot_base::Bounds<essay_plot_base::Canvas>,
        style: &dyn essay_plot_base::StyleOpt,
    ) {
        let vertices = tf32!([
            [0.0, 0.0], [1.0, 0.0], [0.5, 1.0],
            [1.5, 1.0]
            ]);
        let colors: Tensor<u32> = tensor!([0xff0000ff, 0xffff00ff, 0xff0000ff, 0x0080ffff]);
        let triangles: Tensor<u32> = tensor!([[0, 1, 2], [1, 3, 2]]);

        let vertices = to_canvas.transform(&vertices);

        renderer.draw_triangles(vertices, colors, triangles).unwrap();
    }
}