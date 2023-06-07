use core::fmt;
use std::{f32::consts::{PI}};

use essay_tensor::{Tensor, tensor::{Axis, TensorVec}};

use crate::{figure::{Affine2d, Bounds, Data, Point}, driver::{Renderer, Device}};

use super::{ArtistTrait, Path, PathCode};

pub struct Collection {
    lines: Tensor, // 2d tensor representing a graph
    path: Path<Data>,
    clip_bounds: Bounds<Device>,
}

impl Collection {
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
            codes.push(PathCode::MoveTo(Point(xy[0], xy[1])));
            is_active = true;
        } else {
            codes.push(PathCode::LineTo(Point(xy[0], xy[1])));
        }

        // TODO: build new tensor
    }

    Path::new(codes)
}

impl ArtistTrait for Collection {
    fn get_data_bounds(&mut self) -> Bounds<Data> {
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

        gc.color(0x7f3f00ff);
        gc.linewidth(20.);
        
        /*
        let path = Path::closed_poly(tf32!([
            [2., 2.], [8., 2.], [8., 8.], [2., 8.],
        ]));
        */
        
            /*
        let path = Path::closed_poly(tf32!([
            [2., 5.], [4., 7.], [6., 7.],
            [8., 5.], [6., 3.], [4., 3.],
        ]));
        */

        /*
        let path = Path::closed_poly(tf32!([
            [4., 4.], [4., 5.], [5., 5.], 
            [5., 6.], [6., 6.], [6., 5.],
            [7., 5.], [7., 4.], [6., 4.], 
            [6., 3.], [5., 3.], [5., 4.], 
        ]));
        */

        let len = 3;
        let s = 4.0;
        let q = 0.2;
        let delta = PI / len as f32;
        let mut vec = TensorVec::<[f32; 2]>::new();
        for i in 0..len {
            let theta = 2. * i as f32 * delta;

            // flipped sin/cos so theta=0 is up
            vec.push([5. + s * theta.sin(), 5. + s * theta.cos()]);
            vec.push([5. + s * q * (theta + delta).sin(), 5. + s * q * (theta + delta).cos()]);
        }

        let path = Path::closed_poly(vec.into_tensor());

        renderer.draw_path(&gc, &path, to_device, clip).unwrap();
    }
}

impl fmt::Debug for Collection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.lines.dim(0) {
            0 => {
                write!(f, "Collection[]")
            },
            1 => {
                write!(f, "Collection[({}, {})]", self.lines[(0, 0)], self.lines[(0, 1)])
            },
            2 => {
                write!(f, "Collection[({}, {}), ({}, {})]", 
                    self.lines[(0, 0)], self.lines[(0, 1)],
                    self.lines[(1, 0)], self.lines[(1, 1)])
            },
            n => {
                write!(f, "Collection[({}, {}), ({}, {}), ..., ({}, {})]", 
                    self.lines[(0, 0)], self.lines[(0, 1)],
                    self.lines[(1, 0)], self.lines[(1, 1)],
                    self.lines[(n - 1, 0)], self.lines[(n - 1, 1)])
            }
        }
    }
}
