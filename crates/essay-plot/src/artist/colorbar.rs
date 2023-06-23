use essay_plot_base::{Point, Canvas, Bounds, driver::Renderer, Affine2d, Clip, PathOpt, Color};
use essay_tensor::{Tensor, init::linspace, tf32};

use crate::frame::Data;

use super::{Artist, colormesh::ColorMesh, paths, PathStyle};

pub struct Colorbar {
    bounds: Bounds<Data>,
    pos: Bounds<Canvas>,
    mesh: ColorMesh,
    data: Tensor,
}

impl Colorbar {
    pub fn new() -> Self {
        Self {
            bounds: Bounds::zero(),
            pos: Bounds::zero(),
            data: tf32!([0., 1.]),
            mesh: ColorMesh::new(tf32!([[0.]])),
        }
    }

    pub(crate) fn set_pos(&mut self, pos: Bounds<Canvas>) {
        self.pos = pos;
    }
}

impl Artist<Data> for Colorbar {
    fn update(&mut self, canvas: &Canvas) {
        self.bounds = Bounds::new(Point(0., 0.), Point(1., 256.));
        self.data = linspace(0., 1., 256).reshape([256, 1]);
        self.mesh.set_data(self.data.clone());
        self.mesh.update(canvas);
    }

    fn get_extent(&mut self) -> Bounds<Data> {
        self.bounds.clone()
    }

    fn draw(
        &mut self, 
        renderer: &mut dyn Renderer,
        _to_canvas: &Affine2d,
        clip: &Clip,
        style: &dyn PathOpt,
    ) {
        let to_canvas = self.bounds.affine_to(&self.pos);
        // self.mesh.draw(renderer, &to_canvas, clip, style);

        let path = paths::bounds(&self.pos);
        let mut pstyle = PathStyle::new();
        pstyle.face_color(Color(0x0));
        pstyle.edge_color(Color(0xff));
        //pstyle.line_width(2.);
        renderer.draw_path(&path, &pstyle, clip);

        self.mesh.draw(renderer, &to_canvas, clip, style);
    }
}