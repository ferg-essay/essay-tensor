use essay_plot_base::{Canvas, Bounds, Point, Clip, PathOpt, Path};
use essay_tensor::{Tensor, tensor::TensorVec, tf32, math::normalize_unit};

use crate::frame::Data;

use super::{Artist, ColorMap, ColorMaps, PathStyle};


pub struct ColorMesh {
    data: Tensor,
    xy: Tensor,
}

impl ColorMesh {
    pub fn new(data: impl Into<Tensor>) -> Self {
        let data : Tensor = data.into();

        assert!(data.rank() == 2, "colormesh requires 2d value {:?}", data.shape().as_slice());

        Self {
            data,
            xy: Tensor::empty(),
        }
    }

    pub(crate) fn set_data(&mut self, data: Tensor) {
        assert!(data.rank() == 2, "colormesh requires 2d value {:?}", data.shape().as_slice());

        self.data = data;
    }
}

impl Artist<Data> for ColorMesh {
    fn update(&mut self, _canvas: &Canvas) {
        let mut xy = TensorVec::<f32>::new();

        for j in 0..self.data.rows() {
            for i in 0..self.data.cols() {
                xy.push(i as f32);
                xy.push(j as f32);
            }
        }
        let len = xy.len();
        self.xy = xy.into_tensor().reshape([len / 2, 2]);
    }
    
    fn get_extent(&mut self) -> Bounds<Data> {
        Bounds::new(
            Point(0.0, 0.0), 
            Point(self.data.cols() as f32, self.data.rows() as f32)
        )
    }

    fn draw(
        &mut self, 
        renderer: &mut dyn essay_plot_base::driver::Renderer,
        to_canvas: &essay_plot_base::Affine2d,
        clip: &Clip,
        style: &dyn PathOpt,
    ) {
        let path = Path::<Data>::closed_poly(tf32!([
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0],
            [0.0, 1.0]
            ]));
            
        let scale_canvas = to_canvas.strip_translation();
        let path: Path<Canvas> = path.transform(&scale_canvas);
        let xy = to_canvas.transform(&self.xy);

        let norm = normalize_unit(&self.data);

        let colormap : ColorMap = ColorMaps::Default.into();

        let mut colors = TensorVec::<u32>::new();
        for v in norm.iter() {
            colors.push(colormap.map(*v).to_rgba());
        }

        let colors = colors.into_tensor();

        let mut style = PathStyle::new();

        style.edge_color("k");

        renderer.draw_markers(&path, &xy, &tf32!(), &colors, &style, clip).unwrap();
    }
}
