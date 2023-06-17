use essay_plot_base::{Affine2d, Path, Bounds, Canvas, PathOpt, driver::Renderer, JoinStyle, affine};
use essay_tensor::Tensor;
use essay_plot_macro::*;

use crate::{artist::{paths::{self}, Artist, PathCollection, Markers, PathStyle}, graph::{Graph}, frame::{Data}};

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

    ScatterOpt::new(graph.add_plot(plot))
}

#[derive_plot_opt(ScatterOpt)]
pub struct ScatterPlot {
    xy: Tensor,
    collection: PathCollection,
    is_stale: bool,

    #[option]
    style: PathStyle,

    #[option]
    size: f32,

    #[option(Into)]
    marker: Markers,
}

impl ScatterPlot {
    fn new(xy: Tensor) -> Self {
        let scale = 10.;
        let size = scale * scale;
        let path = paths::unit_pos().transform(
            &affine::scale(scale, scale)
        );

        let collection = PathCollection::new(path, xy.clone());
        let mut style = PathStyle::new();

        //style.linewidth(1.5);
        style.joinstyle(JoinStyle::Round);

        Self {
            xy,
            style,
            size,
            marker: Markers::Circle,
            collection,
            is_stale: true,
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
        if self.is_stale {
            self.is_stale = false;

            // 0.5 because source is [-1, 1]
            let scale = 0.5 * self.size.sqrt() * canvas.scale_factor();

            let path: Path<Canvas> = self.marker.get_scaled_path(scale);

            self.collection = PathCollection::new(path, &self.xy);
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
        style: &dyn PathOpt,
    ) {
        let style = self.style.push(style);

        self.collection.draw(renderer, to_canvas, clip, &style)
    }
}
