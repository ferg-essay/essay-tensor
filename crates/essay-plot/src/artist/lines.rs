use core::fmt;

use essay_tensor::{Tensor, tensor::Axis};

use essay_plot_base::{
    Affine2d, Bounds, Point, Canvas, Path, PathCode, PathOpt,
    driver::Renderer, Clip
};

use crate::{
    artist::PathStyle, 
    frame::{Data, LegendHandler}, 
    graph::{ConfigArc},
    data_artist_option_struct, path_style_options
};

use super::{Artist, PlotArtist, PlotId};

#[derive(Clone, PartialEq, Debug)]
pub enum DrawStyle {
    StepsPre,
    StepsMid,
    StepsPost
}

pub struct Lines2d {
    xy: Tensor, // 2d tensor representing a graph
    path: Path<Data>,

    style: PathStyle,
    label: Option<String>,

    extent: Bounds<Data>,
}

impl Lines2d {
    pub fn from_xy(x: impl Into<Tensor>, y: impl Into<Tensor>) -> Self {
        let x = x.into();
        let y = y.into();

        assert_eq!(x.len(), y.len());

        let lines = x.stack(&[y], Axis::axis(-1));

        let path = build_path(&lines);

        Self {
            xy: lines,
            path,
            style: PathStyle::new(),

            label: None,

            extent: Bounds::<Data>::none(),
        }
    }
}

fn build_path(line: &Tensor) -> Path<Data> {
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
        self.extent = Bounds::from(&self.xy);
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

        renderer.draw_path(&path, &style, clip).unwrap();
    }
}

impl PlotArtist<Data> for Lines2d {
    type Opt = LinesOpt;

    fn config(&mut self, cfg: &ConfigArc, id: PlotId) -> Self::Opt {
        self.style = PathStyle::from_config(cfg, "lines");

        unsafe { LinesOpt::new(id) }
    }

    fn get_legend(&self) -> Option<LegendHandler> {
        match &self.label {
            Some(label) => {
                let style = self.style.clone();
                Some(LegendHandler::new(label.clone(), 
                move |renderer, bounds| {
                    let line = Path::<Canvas>::from([
                        [bounds.xmin(), bounds.ymid()],
                        [bounds.xmax(), bounds.ymid()],
                    ]);
                    renderer.draw_path(&line, &style, &Clip::None).unwrap();
                }))
            },
            None => None,
        }
    }
}

data_artist_option_struct!(LinesOpt, Lines2d);

impl LinesOpt {
    path_style_options!(style);

    pub fn label(&mut self, label: &str) -> &mut Self {
        self.write(|artist| {
            if label.len() > 0 {
                artist.label = Some(label.to_string());
            } else {
                artist.label = None;
            }
        });

        self
    }
}

//impl PathStyleArtist for Lines2d {
//    fn style_mut(&mut self) -> &mut PathStyle {
//        &mut self.style
//    }
//}

impl fmt::Debug for Lines2d {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.xy.dim(0) {
            0 => {
                write!(f, "Lines2D[]")
            },
            1 => {
                write!(f, "Lines2D[({}, {})]", self.xy[(0, 0)], self.xy[(0, 1)])
            },
            2 => {
                write!(f, "Lines2D[({}, {}), ({}, {})]", 
                    self.xy[(0, 0)], self.xy[(0, 1)],
                    self.xy[(1, 0)], self.xy[(1, 1)])
            },
            n => {
                write!(f, "Lines2D[({}, {}), ({}, {}), ..., ({}, {})]", 
                    self.xy[(0, 0)], self.xy[(0, 1)],
                    self.xy[(1, 0)], self.xy[(1, 1)],
                    self.xy[(n - 1, 0)], self.xy[(n - 1, 1)])
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