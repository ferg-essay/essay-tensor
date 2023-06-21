use essay_plot_base::{Affine2d, Path, Bounds, Canvas, PathOpt, driver::Renderer, Clip};
use essay_tensor::Tensor;

use crate::{
    artist::{Container, paths, patch::{PathPatch}, Artist}, 
    graph::{Graph, PlotOpt}, 
    frame::{Data}, 
    data_artist_option_struct
};

// self as essay_plot needed for #[derive_plot_opt]
extern crate self as essay_plot;

pub fn bar_y(
    graph: &mut Graph, 
    y: impl Into<Tensor>, 
) -> PlotOpt { // BarOpt {
    let y : Tensor = y.into();

    let plot = BarPlot::new(y);

    //let plot_ref = graph.add_plot_artist(plot);

    //BarOpt::new(plot_ref)
    graph.add_simple_artist(plot)
}

pub struct BarPlot {
    y: Tensor,
    container: Container<Data>,
    is_modified: bool,

    // #[option]
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
            
                self.container.push(PathPatch::<Data>::new(path));
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
        clip: &Clip,
        style: &dyn PathOpt,
    ) {
        self.container.draw(renderer, to_canvas, clip, style)
    }
}

data_artist_option_struct!(BarOpt, BarPlot);

impl BarOpt {
    pub fn width(&mut self, width: f32) -> &mut Self {
        self.write(|bar| bar.width = width);

        self
    }
}

