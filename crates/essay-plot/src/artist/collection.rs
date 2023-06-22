use core::fmt;

use essay_tensor::{Tensor};
use essay_plot_base::{Affine2d, Bounds, Path, PathOpt, Canvas, driver::{Renderer}, Clip};

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
            bounds: Bounds::from(&xy),
            xy,
            color: Tensor::empty(),
            scale: Tensor::empty(),
            style: PathStyle::new(), // needs to be loop
        }
    }
}

impl Artist<Data> for PathCollection {
    fn update(&mut self, _canvas: &Canvas) {
        self.bounds = Bounds::from(&self.xy);
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

        let style = self.style.push(style);

        renderer.draw_markers(&self.path, &xy, &self.scale, &self.color, &style, clip).unwrap();
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
