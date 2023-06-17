use essay_plot_base::{Affine2d, Path, Bounds, Canvas, StyleOpt, driver::Renderer, JoinStyle};
use essay_tensor::Tensor;
use essay_plot_macro::*;

use crate::{artist::{Container, paths::{self, Unit}, patch::{PathPatch}, ArtistStyle, Artist, Collection, Markers, Style, StyleChain}, graph::{Graph}, frame::{Data, ArtistId}};

// self as essay_plot needed for #[derive_plot_opt]
extern crate self as essay_plot;

pub fn scatter(
    graph: &mut Graph, 
    x: impl Into<Tensor>, 
    y: impl Into<Tensor>, 
) -> ScatterOpt {
    let x : Tensor = x.into();
    let y : Tensor = y.into();

    let plot = ScatterPlot::new(x.stack(&[y], -1));

    let plot_ref = graph.add_plot(plot);

    ScatterOpt::new(plot_ref)
}

#[derive_plot_opt(ScatterOpt)]
pub struct ScatterPlot {
    xy: Tensor,
    collection: Collection,
    is_modified: bool,

    style: Style,

    #[option]
    size: f32,

    #[option(Into)]
    marker: Markers,
}

impl ScatterPlot {
    fn new(xy: Tensor) -> Self {
        let scale = 10.;
        let size = scale * scale;
        let path = paths::unit_pos().transform(&Affine2d::eye().scale(scale, scale));

        let collection = Collection::new(path, xy.clone());
        let mut style = Style::new();

        style.linewidth(1.5);
        style.joinstyle(JoinStyle::Miter);

        Self {
            xy,
            style,
            size,
            marker: Markers::Circle,
            collection,
            is_modified: true,
        }
    }

    fn size(&mut self, size: f32) -> &mut Self {
        assert!(size >= 0.);

        self.size = size;

        self
    }

    fn marker(&mut self, marker: impl Into<Markers>) -> &mut Self {
        self.marker = marker.into();

        self
    }
}

impl Artist<Data> for ScatterPlot {
    fn update(&mut self, canvas: &Canvas) {
        if self.is_modified {
            self.is_modified = false;

            // 0.5 because source is [-1, 1]
            let scale = 0.5 * self.size.sqrt() * canvas.scale_factor();

            let path: Path<Canvas> = self.marker.get_scaled_path(scale);

            self.collection = Collection::new(path, &self.xy);
        }

        self.collection.update(canvas);
    }

    fn get_extent(&mut self) -> Bounds<Data> {
        self.collection.get_extent()
    }

    fn draw(
        &mut self, 
        renderer: &mut dyn Renderer,
        to_canvas: &Affine2d,
        clip: &Bounds<Canvas>,
        style: &dyn StyleOpt,
    ) {
        let style = StyleChain::new(style, &self.style);

        self.collection.draw(renderer, to_canvas, clip, &style)
    }
}
