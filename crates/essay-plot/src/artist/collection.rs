use core::fmt;

use essay_tensor::{Tensor, tensor::{Axis}};
use essay_plot_base::{Affine2d, Bounds, Path, PathCode, StyleOpt, Point, Canvas, Style, driver::{Renderer}};

use crate::frame::Data;

use super::{ArtistTrait};

pub struct Collection {
    path: Path<Canvas>,
    offsets: Tensor, // 2d tensor representing a graph
    style: Style,
    bounds: Bounds<Data>,
}

impl Collection {
    pub fn new(path: Path<Canvas>, xy: impl Into<Tensor>) -> Self {
        let xy = xy.into();

        assert!(xy.cols() == 2, "Collection requires 2 column data");

        Self {
            path,
            bounds: Bounds::from(xy.clone()), // replace clone with &ref
            offsets: xy,
            style: Style::new(), // needs to be loop
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

    fn style_mut(&mut self) -> &mut Style {
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

impl ArtistTrait<Data> for Collection {
    fn get_extent(&mut self) -> Bounds<Data> {
        let mut bounds = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];

        for point in self.offsets.iter_slice() {
            bounds[0] = f32::min(bounds[0], point[0]);
            bounds[1] = f32::min(bounds[1], point[1]);
            bounds[2] = f32::max(bounds[2], point[0]);
            bounds[3] = f32::max(bounds[3], point[1]);
        }

        Bounds::from(bounds)
    }

    fn draw(
        &mut self, 
        renderer: &mut dyn Renderer, 
        to_canvas: &Affine2d,
        clip: &Bounds<Canvas>,
        style: &dyn StyleOpt,
    ) {
        //let mut gc = renderer.new_gc();

        //gc.color(0x7f3f00ff);
        //gc.linewidth(10.);

        let xy = to_canvas.transform(&self.offsets);

        renderer.draw_markers(&self.path, &xy, style, clip).unwrap();
    }
}

impl fmt::Debug for Collection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.offsets.dim(0) {
            0 => {
                write!(f, "Collection[]")
            },
            1 => {
                write!(f, "Collection[({}, {})]", self.offsets[(0, 0)], self.offsets[(0, 1)])
            },
            2 => {
                write!(f, "Collection[({}, {}), ({}, {})]", 
                    self.offsets[(0, 0)], self.offsets[(0, 1)],
                    self.offsets[(1, 0)], self.offsets[(1, 1)])
            },
            n => {
                write!(f, "Collection[({}, {}), ({}, {}), ..., ({}, {})]", 
                    self.offsets[(0, 0)], self.offsets[(0, 1)],
                    self.offsets[(1, 0)], self.offsets[(1, 1)],
                    self.offsets[(n - 1, 0)], self.offsets[(n - 1, 1)])
            }
        }
    }
}
