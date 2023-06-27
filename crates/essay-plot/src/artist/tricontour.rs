use essay_plot_base::{Canvas, Bounds, Point, Clip, PathOpt, Path};
use essay_tensor::{Tensor, tensor::TensorVec, tf32, math::normalize_unit};

use crate::{frame::Data, contour::{ContourGenerator, TriContourGenerator}, tri::Triangulation};

use super::{Artist, ColorMap, ColorMaps, PathStyle};

pub struct Level {
    value: f32,
    paths: Vec<Path<Data>>,
}

impl Level {
    fn new(value: f32, paths: Vec<Path<Data>>) -> Self {
        Self {
            value,
            paths,
        }
    }
}

pub struct TriContour {
    data: Tensor,
    color_map: ColorMap,

    tri: Triangulation,
    norm: Tensor,
    levels: Vec<Level>,
    bounds: Bounds<Data>,
}

impl TriContour {
    pub fn new(tri: impl Into<Triangulation>, data: impl Into<Tensor>) -> Self {
        let tri: Triangulation = tri.into();
        let data : Tensor = data.into();

        assert!(data.rank() == 1, "contour requires 1d value {:?}", data.shape().as_slice());

        Self {
            data,
            tri,
            norm: Tensor::empty(),
            color_map: ColorMaps::Default.into(),
            bounds: Bounds::zero(),
            levels: Vec::new(),
        }
    }

    pub(crate) fn set_data(&mut self, data: Tensor) {
        assert!(data.rank() == 2, "contour requires 2d value {:?}", data.shape().as_slice());

        self.data = data;
    }
}

impl Artist<Data> for TriContour {
    fn update(&mut self, _canvas: &Canvas) {
        let (rows, cols) = (self.data.rows(), self.data.cols());

        //for vert in self.tri.triangles().iter_slice() {
        //  xy.push([i as f32, j as f32]);
        //}

        self.bounds = Bounds::<Data>::from(self.tri.vertices());

        self.norm = normalize_unit(&self.data);

        let mut cg = TriContourGenerator::new(&self.tri, self.data.clone());

        let level_thresholds = vec![
            -1.5,
            -1.,
            -0.5, 
            0., 
            0.5,
            1.,
            1.5,
            ];
        let mut levels = Vec::<Level>::new();

        for threshold in &level_thresholds {
            let paths = cg.contour_lines(*threshold);

            let paths: Vec<Path<Data>> = paths.iter()
                .map(|p| Path::<Data>::lines(p))
                .collect();

            levels.push(Level::new(*threshold, paths));
        }

        self.levels = levels;

        // self.xy = xy.into_tensor();
    }
    
    fn get_extent(&mut self) -> Bounds<Data> {
        self.bounds.clone()
    }

    fn draw(
        &mut self, 
        renderer: &mut dyn essay_plot_base::driver::Renderer,
        to_canvas: &essay_plot_base::Affine2d,
        clip: &Clip,
        _style: &dyn PathOpt,
    ) {
        let mut style = PathStyle::new();

        style.edge_color("k");
        style.line_width(1.);

        //renderer.draw_markers(&path, &xy, &tf32!(), &colors, &style, clip).unwrap();

        for level in &self.levels {
            for path in &level.paths {
                let path : Path<Canvas> = path.transform(&to_canvas);

                renderer.draw_path(&path, &style, clip).unwrap();
            }
        }
    }
}
