use core::fmt;
use std::cmp;

use essay_tensor::{Tensor, tensor::Axis};

use crate::{axes::{Rect, Affine2d}, device::Renderer};

use super::Artist;

pub struct Lines2D {
    lines: Tensor, // 2d tensor representing a graph
    data_box: Rect, // bounding box
}

impl Lines2D {
    pub fn from_xy(x: impl Into<Tensor>, y: impl Into<Tensor>) -> Self {
        let x = x.into();
        let y = y.into();

        assert_eq!(x.len(), y.len());

        let lines = x.stack(&[y], Axis::axis(-1));

        let mut bounds = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];

        for point in lines.iter_slice() {
            bounds[0] = f32::min(bounds[0], point[0]);
            bounds[1] = f32::min(bounds[1], point[1]);
            bounds[2] = f32::max(bounds[2], point[0]);
            bounds[3] = f32::max(bounds[3], point[1]);
        }

        Self {
            lines,
            data_box: Rect::new(
                bounds[0],
                bounds[1], 
                bounds[2] - bounds[0], 
                bounds[3] - bounds[1]
            ),
        }
    }
}

impl Artist for Lines2D {
    fn draw(&mut self, renderer: &mut dyn Renderer) {
        let transform = Affine2d::eye();

        todo!()
    }
}

impl fmt::Debug for Lines2D {
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

    use super::Lines2D;

    #[test]
    fn test_lines() {
        let lines = Lines2D::from_xy(
            tf32!([1., 2., 4., 8.]),
            tf32!([10., 20., 40., 80.])
        );
        println!("Lines {:?}", &lines);
    }
}