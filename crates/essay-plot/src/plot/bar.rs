use essay_plot_base::{Affine2d, Path, Bounds, Canvas, PathOpt, driver::Renderer};
use essay_tensor::Tensor;
use essay_plot_macro::*;

use crate::{artist::{Container, paths, patch::{PathPatch}, ArtistStyle, Artist}, graph::{Graph}, frame::{Data, ArtistId}};

// self as essay_plot needed for #[derive_plot_opt]
extern crate self as essay_plot;

pub fn bar_y(
    graph: &mut Graph, 
    y: impl Into<Tensor>, 
) -> BarOpt {
    let y : Tensor = y.into();

    let plot = BarPlot::new(y);

    let plot_ref = graph.add_plot(plot);

    BarOpt::new(plot_ref)
}

#[derive_plot_opt(BarOpt)]
pub struct BarPlot {
    y: Tensor,
    container: Container<Data>,
    is_modified: bool,

    #[option]
    width: f32,
}

impl BarPlot {
    fn new(y: Tensor) -> Self {
        let width = 1.;
        let container = Container::new();

        Self {
            width,
            y,
            container,
            is_modified: true,
        }
    }

    fn width(&mut self, width: f32) -> &mut Self {
        self.width = width;

        self
    }
}

impl Artist<Data> for BarPlot {
    fn update(&mut self, canvas: &Canvas) {
        if self.is_modified {
            self.is_modified = false;

            self.container.clear();

            for (i, value) in self.y.iter().enumerate() {
                let scale = Affine2d::eye()
                    .scale(self.width, *value)
                    .translate(i as f32 - self.width * 0.5, 0.);
                let path: Path<Data> = paths::unit_pos().transform(&scale);
            
                let id = ArtistId::new(0);
                self.container.push(ArtistStyle::new(id, PathPatch::<Data>::new(path)));
            }
        }

        self.container.update(canvas);
    }

    fn get_extent(&mut self) -> Bounds<Data> {
        self.container.get_extent()
    }

    fn draw(
        &mut self, 
        renderer: &mut dyn Renderer,
        to_canvas: &Affine2d,
        clip: &Bounds<Canvas>,
        style: &dyn PathOpt,
    ) {
        self.container.draw(renderer, to_canvas, clip, style)
    }
}
