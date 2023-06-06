use core::fmt;
use std::cmp;

use essay_tensor::{Tensor, tensor::Axis};

use crate::{
    figure::{Rect, Affine2d, Bounds, Data}, 
    driver::{Renderer, Device}
};

use super::{Artist, Path, PathCode};

pub struct Lines2d {
    lines: Tensor, // 2d tensor representing a graph
    path: Path<Data>,
    clip_bounds: Bounds<Device>,
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
            clip_bounds: Bounds::<Device>::none(),
        }
    }
}

fn build_path(line: &Tensor, xmin: f32, xmax: f32) -> Path<Data> {
    let mut codes = Vec::<PathCode>::new();
    
    let mut is_active = false;
    for xy in line.iter_slice() {
        if ! is_active {
            codes.push(PathCode::MoveTo);
            is_active = true;
        } else {
            codes.push(PathCode::LineTo);
        }

        // TODO: build new tensor
    }

    Path::new(line, codes)
}

impl Artist for Lines2d {
    fn get_data_bounds(&self) -> Bounds<Data> {
        let mut bounds = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];

        for point in self.lines.iter_slice() {
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
        to_device: &Affine2d,
        clip: &Bounds<Device>,
    ) {
        let mut gc = renderer.new_gc();

        gc.rgba(0x7f3f00ff);
        gc.linewidth(20.);

        renderer.draw_path(&gc, &self.path, to_device, clip).unwrap();
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