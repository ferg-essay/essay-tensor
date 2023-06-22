use essay_tensor::Tensor;

use crate::{graph::Graph, artist::{Artist, PathStyle, ColorMap}};
use essay_plot_base::{Bounds, Point, Canvas, Clip, PathOpt, Path, Color};
use essay_tensor::{tf32, tensor::{self, TensorVec}, math::normalize_unit};

use crate::frame::Data;

pub fn pcolormesh(
    graph: &mut Graph, 
    data: impl Into<Tensor>,
) {
    let pcolor = PColorMesh::new(data);
    
    graph.add_simple_artist(pcolor);
}

pub struct PColorMesh {
    data: Tensor,
    xy: Tensor,
}

impl PColorMesh {
    pub fn new(data: impl Into<Tensor>) -> Self {
        let data : Tensor = data.into();

        assert!(data.rank() == 2, "pcolor requires 2d value {:?}", data.shape().as_slice());

        Self {
            data,
            xy: Tensor::empty(),
        }
    }
}

impl Artist<Data> for PColorMesh {
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
            
        // path needs to be scaled to canvas but not translated.
        // TODO: add a strip_translation to Affine2d?
        let scale_canvas = to_canvas.strip_translation();
        let path: Path<Canvas> = path.transform(&scale_canvas);
        let xy = to_canvas.transform(&self.xy);

        let norm = normalize_unit(&self.data);
        let c = Color::from(0xf9e300).closest_name();
        
        //let colormap = ColorMap::from(["red", "white", "blue"]);
        let colormap = ColorMap::from(["#ff0000", "white", "#0000ff"]);
        //let colormap = ColorMap::from(["red", "green", "pink", "yellow"]);

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
    }
}
