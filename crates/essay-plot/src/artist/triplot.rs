use essay_plot_base::{Bounds, Point, Canvas, Clip, PathOpt, Path, Color, PathCode};
use essay_tensor::{tf32, tensor::{self, TensorVec}, Tensor, math::normalize_unit};

use crate::{frame::Data, tri::{Triangulation, triangulate}};

use super::{Artist, PathStyle, colormap::ColorMap};

pub struct TriPlot {
    data: Tensor,
    triangulation: Option<Triangulation>,
    is_stale: bool,
}

impl TriPlot {
    pub fn new(data: impl Into<Tensor>) -> Self {
        let data : Tensor = data.into();

        assert!(data.rank() == 2, "triplot requires 2d value {:?}", data.shape().as_slice());
        assert!(data.cols() == 2, "triplot requires 2d value {:?}", data.shape().as_slice());

        Self {
            data,
            triangulation: None,
            is_stale: true,
        }
    }
}

impl Artist<Data> for TriPlot {
    fn update(&mut self, _canvas: &Canvas) {
        if self.is_stale {
            self.is_stale = false;
            self.triangulation = Some(triangulate(&self.data));
        }
    }
    
    fn get_extent(&mut self) -> Bounds<Data> {
        Bounds::from(&self.data)
    }

    fn draw(
        &mut self, 
        renderer: &mut dyn essay_plot_base::driver::Renderer,
        to_canvas: &essay_plot_base::Affine2d,
        clip: &Clip,
        style: &dyn PathOpt,
    ) {
        if let Some(tri) = &self.triangulation {
            let mut codes = Vec::<PathCode>::new();

            let xy = tri.vertices();
            for edge in tri.edges().iter_slice() {
                let (x0, y0) = (xy[(edge[0], 0)], xy[(edge[0], 1)]);
                let (x1, y1) = (xy[(edge[1], 0)], xy[(edge[1], 1)]);

                codes.push(PathCode::MoveTo(Point(x0, y0)));
                codes.push(PathCode::LineTo(Point(x1, y1)));
            
            }

            let path = Path::<Data>::new(codes);

        /*
            
        // path needs to be scaled to canvas but not translated.
        // TODO: add a strip_translation to Affine2d?
        let scale_canvas = to_canvas.strip_translation();
        let path: Path<Canvas> = path.transform(&scale_canvas);
        let xy = to_canvas.transform(&self.xy);

        let norm = normalize_unit(&self.data);

        let colormap = ColorMap::from([(0., Color(0xffff00ff)), (1., Color(0x000080ff))]);

        let mut colors = TensorVec::<u32>::new();
        for v in norm.iter() {
            colors.push(colormap.map(*v).to_rgba());
        }
        let colors = colors.into_tensor();
        //let colors: Tensor<u32> = tensor!([0xff0000ff, 0xffff00ff, 0xff0000ff, 0x0080ffff]);
        //let triangles: Tensor<u32> = tensor!([[0, 1, 2], [1, 3, 2]]);
        let mut style = PathStyle::new();
        //style.color("xkcd:purple");
        style.edge_color("k");

        renderer.draw_markers(&path, &xy, &tf32!(), &colors, &style, clip).unwrap();
        */
            let path = path.transform(to_canvas);

            renderer.draw_path(&path, style, clip).unwrap();
        }
    }
}
