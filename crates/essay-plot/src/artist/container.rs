use essay_plot_base::{
    Bounds, Affine2d, Point, Coord, Canvas, PathOpt,
    driver::{Renderer}
};

use crate::{graph::{ConfigArtist, PlotOpt, Config, PlotId, ConfigArc, PathStyleArtist}, frame::Data};

use super::{ArtistStyle, Artist, PathStyle};

pub struct Container<M: Coord> {
    artists: Vec<ArtistStyle<M>>,
    style: PathStyle,
}

impl<M: Coord> Container<M> {
    pub fn new() -> Self {
        Self {
            artists: Vec::new(),
            style: PathStyle::new(),
        }
    }

    fn style_mut(&mut self) -> &mut PathStyle {
        &mut self.style
    }

    pub fn push(&mut self, artist: ArtistStyle<M>) {
        self.artists.push(artist);
    }

    pub(crate) fn clear(&mut self) {
        self.artists.drain(..);
    }
}

impl<M: Coord> Artist<M> for Container<M> {
    fn update(&mut self, canvas: &Canvas) {
        for artist in &mut self.artists {
            artist.update(canvas);
        }
    }

    fn get_extent(&mut self) -> Bounds<M> {
        let mut bounds = Bounds::<M>::none();

        for artist in &mut self.artists {
            bounds = if bounds.is_none() {
                    artist.get_extent().clone()
            } else {
                bounds.union(&artist.get_extent())
            };
        }

        bounds
    }

    fn draw(
        &mut self, 
        renderer: &mut dyn Renderer,
        to_device: &Affine2d,
        clip: &Bounds<Canvas>,
        style: &dyn PathOpt,
    ) {
        //let style_cycle = Style::new();
        //let style = Style::chain(style, &style_cycle);

        for artist in &mut self.artists {
            artist.draw(renderer, to_device, clip, style);
        }
    }
}

impl ConfigArtist<Data> for Container<Data> {
    type Opt = PlotOpt;

    fn config(&mut self, _cfg: &ConfigArc, id: PlotId) -> Self::Opt {
        PlotOpt::new::<Container<Data>>(id)
    }
}

impl PathStyleArtist for Container<Data> {
    fn style_mut(&mut self) -> &mut PathStyle {
        &mut self.style
    }
}
