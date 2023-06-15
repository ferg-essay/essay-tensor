use essay_plot_base::{Affine2d, Path, Bounds, Color, Canvas};
use essay_tensor::Tensor;

use crate::{artist::{Container, paths, patch::{PathPatch}, ArtistStyle, Artist}, graph::{Graph, PlotOpt, PlotRef}, frame::{Data, ArtistId}};

pub fn bar_y(
    graph: &mut Graph, 
    y: impl Into<Tensor>, 
) -> BarOpt {
    let y : Tensor = y.into();

    /*
    let mut container = Container::new();
    let width = 1.;

    for (i, value) in y.iter().enumerate() {
        let scale = Affine2d::eye()
            .scale(width, *value)
            .translate(i as f32 - 0.4, 0.);
        let path: Path<Data> = paths::unit().transform(&scale);
        
        let id = ArtistId::new(0);
        container.push(ArtistStyle::new(id, PathPatch::<Data>::new(path)));
    }
    */

    let plot = BarPlot::new(y);
    //self.artist(lines)
    // graph.add_data_artist(plot)
    let plot_ref = graph.add_plot(plot);

    BarOpt::new(plot_ref)
}
// #[derive_opt(BarOpt)]
pub struct BarPlot {
    width: f32,
    y: Tensor,
    container: Container<Data>,
    is_modified: bool,
}

impl BarPlot {
    fn new(y: Tensor) -> Self {
        let width = 1.;
        let mut container = Container::new();

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
                    .translate(i as f32 - 0.4, 0.);
                let path: Path<Data> = paths::unit().transform(&scale);
            
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
        renderer: &mut dyn essay_plot_base::driver::Renderer,
        to_canvas: &Affine2d,
        clip: &essay_plot_base::Bounds<essay_plot_base::Canvas>,
        style: &dyn essay_plot_base::StyleOpt,
    ) {
        self.container.draw(renderer, to_canvas, clip, style)
    }
}

pub struct BarOpt {
    plot: PlotRef<Data, BarPlot>,
}


impl BarOpt {
    fn new(plot: PlotRef<Data, BarPlot>) -> Self {
        Self {
            plot,
        }
    }

    pub fn edgecolor(&mut self, color: impl Into<Color>) -> &mut Self {
        self.plot.write_style(|s| { s.edgecolor(color); });

        self
    }

    pub fn facecolor(&mut self, color: impl Into<Color>) -> &mut Self {
        self.plot.write_style(|s| { s.facecolor(color); });

        self
    }

    pub fn color(&mut self, color: impl Into<Color>) -> &mut Self {
        self.plot.write_style(|s| { s.color(color); });

        self
    }

    pub fn width(&mut self, width: f32) -> &mut Self {
        self.plot.write_artist(|p| { p.width(width); });

        self
    }
}
