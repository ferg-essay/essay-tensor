use essay_plot_base::{
    Coord, Bounds, Affine2d, Canvas, PathOpt,
    driver::Renderer, Clip,
};

use crate::{graph::{ConfigArc}, frame::{LayoutArc, ArtistId, LegendHandler}};

pub trait Artist<M: Coord> {
    fn update(&mut self, canvas: &Canvas);

    fn get_extent(&mut self) -> Bounds<M>;
    
    fn draw(
        &mut self, 
        renderer: &mut dyn Renderer,
        to_canvas: &Affine2d,
        clip: &Clip,
        style: &dyn PathOpt,
    );
}

pub trait PlotArtist<M: Coord> : Artist<M> {
    type Opt;
    
    fn config(
        &mut self, 
        cfg: &ConfigArc, 
        id: PlotId,
    ) -> Self::Opt;

    fn get_legend(&self) -> Option<LegendHandler>;
}

pub trait SimpleArtist<M: Coord> : Artist<M> {
}

pub struct PlotId {
    layout: LayoutArc,
    artist_id: ArtistId,
}

impl PlotId {
    pub(crate) fn new(
        layout: LayoutArc, 
        artist_id: ArtistId
    ) -> Self {
        Self {
            layout,
            artist_id
        }
    }

    pub fn layout(&self) -> &LayoutArc {
        &self.layout
    }

    pub fn id(&self) -> &ArtistId {
        &self.artist_id
    }
}
