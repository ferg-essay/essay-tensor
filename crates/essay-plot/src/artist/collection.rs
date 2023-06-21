use core::fmt;

use essay_tensor::{Tensor, tensor::{Axis}, tf32};
use essay_plot_base::{Affine2d, Bounds, Path, PathCode, PathOpt, Point, Canvas, driver::{Renderer}, Clip};

use crate::{frame::Data, artist::PathStyle};

use super::{Artist};

///
/// Collection of a single path displayed at multiple locations with optional
/// colors and sizes
/// 
pub struct PathCollection {
    path: Path<Canvas>,
    xy: Tensor, // 2d tensor representing a graph
    color: Tensor<u32>,
    scale: Tensor,
    style: PathStyle,
    bounds: Bounds<Data>,
}

impl PathCollection {
    pub fn new(path: Path<Canvas>, xy: impl Into<Tensor>) -> Self {
        let xy = xy.into();

        assert!(xy.cols() == 2, "Collection requires two-column data [x, y]*");

        Self {
            path,
            bounds: Bounds::from(xy.clone()), // replace clone with &ref
            xy,
            color: Tensor::empty(),
            scale: Tensor::empty(),
            style: PathStyle::new(), // needs to be loop
        }
    }

    pub fn from_xy(x: impl Into<Tensor>, y: impl Into<Tensor>) -> Self {
        let x = x.into();
        let y = y.into();

        assert_eq!(x.len(), y.len());

        let lines = x.stack(&[y], Axis::axis(-1));

        let path = build_path(&lines, f32::MIN, f32::MAX);
        /*
        Self {
            offsets: lines,
            path,
            style: Style::new(),
            clip_bounds: Bounds::<Canvas>::none(),
        }
        */
        todo!()
    }

    fn style_mut(&mut self) -> &mut PathStyle {
        &mut self.style
    }
}

fn build_path(line: &Tensor, xmin: f32, xmax: f32) -> Path<Data> {
    let mut codes = Vec::<PathCode>::new();
    
    let mut is_active = false;
    for xy in line.iter_slice() {
        if ! is_active {
            codes.push(PathCode::MoveTo(Point(xy[0], xy[1])));
            is_active = true;
        } else {
            codes.push(PathCode::LineTo(Point(xy[0], xy[1])));
        }

        // TODO: build new tensor
    }

    Path::new(codes)
}

impl Artist<Data> for PathCollection {
    fn update(&mut self, _canvas: &Canvas) {
        let mut bounds = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];

        for point in self.xy.iter_slice() {
            bounds[0] = f32::min(bounds[0], point[0]);
            bounds[1] = f32::min(bounds[1], point[1]);
            bounds[2] = f32::max(bounds[2], point[0]);
            bounds[3] = f32::max(bounds[3], point[1]);
        }

        self.bounds = Bounds::from(bounds);
    }
    
    fn get_extent(&mut self) -> Bounds<Data> {
        self.bounds.clone()
    }

    fn draw(
        &mut self, 
        renderer: &mut dyn Renderer, 
        to_canvas: &Affine2d,
        clip: &Clip,
        style: &dyn PathOpt,
    ) {
        let xy = to_canvas.transform(&self.xy);

        renderer.draw_markers(&self.path, &xy, &self.scale, &self.color, style, clip).unwrap();
    }
}

impl fmt::Debug for PathCollection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.xy.dim(0) {
            0 => {
                write!(f, "Collection[]")
            },
            1 => {
                write!(f, "Collection[({}, {})]", self.xy[(0, 0)], self.xy[(0, 1)])
            },
            2 => {
                write!(f, "Collection[({}, {}), ({}, {})]", 
                    self.xy[(0, 0)], self.xy[(0, 1)],
                    self.xy[(1, 0)], self.xy[(1, 1)])
            },
            n => {
                write!(f, "Collection[({}, {}), ({}, {}), ..., ({}, {})]", 
                    self.xy[(0, 0)], self.xy[(0, 1)],
                    self.xy[(1, 0)], self.xy[(1, 1)],
                    self.xy[(n - 1, 0)], self.xy[(n - 1, 1)])
            }
        }
    }
}
