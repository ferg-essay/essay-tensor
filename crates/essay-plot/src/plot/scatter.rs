use essay_plot_base::{Affine2d, Path, Bounds, Canvas, PathOpt, driver::Renderer, JoinStyle, affine, Clip};
use essay_tensor::Tensor;

use crate::{
    artist::{paths::{self}, Artist, PathCollection, Markers, PathStyle, PlotId, PlotArtist}, 
    graph::{Graph, ConfigArc}, frame::{Data, LegendHandler}, 
    data_artist_option_struct, path_style_options
};

pub fn scatter(
    graph: &mut Graph, 
    x: impl Into<Tensor>, 
    y: impl Into<Tensor>, 
) -> ScatterOpt {
    let x : Tensor = x.into();
    let y : Tensor = y.into();

    let plot = ScatterPlot::new(x.stack(&[y], -1));

    graph.add_plot_artist(plot)
}

pub struct ScatterPlot {
    xy: Tensor,
    collection: PathCollection,
    is_stale: bool,

    style: PathStyle,

    size: f32,

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
        style.join_style(JoinStyle::Round);

        Self {
            xy,
            style,
            size,
            marker: Markers::Circle,
            collection,
            is_stale: true,
        }
    }

    /*
    fn size(&mut self, size: f32) -> &mut Self {
        assert!(size >= 0.);

        self.size = size;

        self
    }

    fn marker(&mut self, marker: impl Into<Markers>) -> &mut Self {
        self.marker = marker.into();

        self
    }
    */
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
        clip: &Clip,
        style: &dyn PathOpt,
    ) {
        let style = self.style.push(style);

        self.collection.draw(renderer, to_canvas, clip, &style)
    }
}

impl PlotArtist<Data> for ScatterPlot {
    type Opt = ScatterOpt;

    fn config(
        &mut self, 
        cfg: &ConfigArc, 
        id: PlotId,
    ) -> Self::Opt {
        self.style = PathStyle::from_config(cfg, "scatter");

        unsafe { ScatterOpt::new(id) }
    }

    fn get_legend(&self) -> Option<LegendHandler> {
        None
    }
}

data_artist_option_struct!(ScatterOpt, ScatterPlot);

impl ScatterOpt {
    path_style_options!(style);

    pub fn marker(&mut self, marker: impl Into<Markers>) -> &mut Self {
        self.write(|plot| plot.marker = marker.into());

        self
    }
}
