use essay_plot_base::{Canvas, Bounds, Point, Clip, PathOpt, Path};
use essay_tensor::{Tensor, tensor::TensorVec, tf32, math::normalize_unit};

use crate::{frame::Data, contour::ContourGenerator};

use super::{Artist, ColorMap, ColorMaps, PathStyle};

pub enum Shading {
    Solid,
    Gouraud,
}

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

pub struct Contour {
    data: Tensor,
    color_map: ColorMap,

    xy: Tensor,
    norm: Tensor,
    levels: Vec<Level>,
}

impl Contour {
    pub fn new(data: impl Into<Tensor>) -> Self {
        let data : Tensor = data.into();

        assert!(data.rank() == 2, "contour requires 2d value {:?}", data.shape().as_slice());

        Self {
            data,
            xy: Tensor::empty(),
            norm: Tensor::empty(),
            color_map: ColorMaps::Default.into(),
            levels: Vec::new(),
        }
    }

    pub(crate) fn set_data(&mut self, data: Tensor) {
        assert!(data.rank() == 2, "contour requires 2d value {:?}", data.shape().as_slice());

        self.data = data;
    }
}

impl Artist<Data> for Contour {
    fn update(&mut self, _canvas: &Canvas) {
        let mut xy = TensorVec::<[f32; 2]>::new();
        let (rows, cols) = (self.data.rows(), self.data.cols());

        for j in 0..rows {
            for i in 0..cols {
                xy.push([i as f32, j as f32]);
            }
        }

        self.norm = normalize_unit(&self.data);

        let mut cg = ContourGenerator::new(self.data.clone());

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

        self.xy = xy.into_tensor();
    }
    
    fn get_extent(&mut self) -> Bounds<Data> {
        let (rows, cols) = (self.data.rows(), self.data.cols());

        Bounds::new(
            Point(0.0, 0.0), 
            Point(cols as f32, rows as f32)
        )
    }

    fn draw(
        &mut self, 
        renderer: &mut dyn essay_plot_base::driver::Renderer,
        to_canvas: &essay_plot_base::Affine2d,
        clip: &Clip,
        _style: &dyn PathOpt,
    ) {
        let path = Path::<Data>::closed_poly(tf32!([
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0],
            [0.0, 1.0]
            ]));
            
        let scale_canvas = to_canvas.strip_translation();
        let path: Path<Canvas> = path.transform(&scale_canvas);
        let xy = to_canvas.transform(&self.xy);

        let colormap = &self.color_map;

        let mut colors = TensorVec::<u32>::new();
        for v in self.norm.iter() {
            colors.push(colormap.map(*v).to_rgba());
        }

        let colors = colors.into_tensor();

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
