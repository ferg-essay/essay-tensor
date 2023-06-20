use core::fmt;

use essay_tensor::{Tensor, tensor::Axis};

use essay_plot_base::{
    Affine2d, Bounds, Point, Canvas, Path, PathCode, PathOpt,
    driver::Renderer, Clip
};

use crate::{frame::Data, artist::PathStyle, graph::{ConfigArtist, PlotOpt, PlotId, ConfigArc, PathStyleArtist}};

use super::{Artist};

#[derive(Clone, PartialEq, Debug)]
pub enum DrawStyle {
    StepsPre,
    StepsMid,
    StepsPost
}

pub struct Lines2d {
    lines: Tensor, // 2d tensor representing a graph
    path: Path<Data>,
    style: PathStyle,
    extent: Bounds<Data>,
}

impl Lines2d {
    pub fn from_xy(x: impl Into<Tensor>, y: impl Into<Tensor>) -> Self {
        let x = x.into();
        let y = y.into();

        assert_eq!(x.len(), y.len());

        let lines = x.stack(&[y], Axis::axis(-1));

        let path = build_path(&lines, f32::MIN, f32::MAX);

        Self {
            lines,
            path,
            style: PathStyle::new(),
            extent: Bounds::<Data>::none(),
        }
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

impl Artist<Data> for Lines2d {
    fn update(&mut self, _canvas: &Canvas) {
        let mut bounds = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];

        for point in self.lines.iter_slice() {
            bounds[0] = f32::min(bounds[0], point[0]);
            bounds[1] = f32::min(bounds[1], point[1]);
            bounds[2] = f32::max(bounds[2], point[0]);
            bounds[3] = f32::max(bounds[3], point[1]);
        }

        self.extent = Bounds::from(bounds)
    }
    
    fn get_extent(&mut self) -> Bounds<Data> {
        self.extent.clone()
    }

    fn draw(
        &mut self, 
        renderer: &mut dyn Renderer, 
        to_canvas: &Affine2d,
        clip: &Clip,
        style: &dyn PathOpt,
    ) {
        let path = self.path.transform(&to_canvas);

        let style = self.style.push(style);

        renderer.draw_path(&style, &path, to_canvas, clip).unwrap();
    }
}

impl ConfigArtist<Data> for Lines2d {
    type Opt = PlotOpt;

    fn config(&mut self, cfg: &ConfigArc, id: PlotId) -> Self::Opt {
        self.style = PathStyle::from_config(cfg, "lines");

        PlotOpt::new::<Self>(id)
    }
}

impl PathStyleArtist for Lines2d {
    fn style_mut(&mut self) -> &mut PathStyle {
        &mut self.style
    }
}

impl fmt::Debug for Lines2d {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.lines.dim(0) {
            0 => {
                write!(f, "Lines2D[]")
            },
            1 => {
                write!(f, "Lines2D[({}, {})]", self.lines[(0, 0)], self.lines[(0, 1)])
            },
            2 => {
                write!(f, "Lines2D[({}, {}), ({}, {})]", 
                    self.lines[(0, 0)], self.lines[(0, 1)],
                    self.lines[(1, 0)], self.lines[(1, 1)])
            },
            n => {
                write!(f, "Lines2D[({}, {}), ({}, {}), ..., ({}, {})]", 
                    self.lines[(0, 0)], self.lines[(0, 1)],
                    self.lines[(1, 0)], self.lines[(1, 1)],
                    self.lines[(n - 1, 0)], self.lines[(n - 1, 1)])
            }
        }
    }
}

#[cfg(test)]
mod test {
    use essay_tensor::prelude::*;

    use super::Lines2d;

    #[test]
    fn test_lines() {
        let lines = Lines2d::from_xy(
            tf32!([1., 2., 4., 8.]),
            tf32!([10., 20., 40., 80.])
        );
        println!("Lines {:?}", &lines);
    }
}