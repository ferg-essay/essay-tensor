use essay_plot_base::{
    Bounds, Affine2d, Coord, Canvas, PathOpt,
    driver::{Renderer}, Clip
};

use crate::{graph::{PlotArtist, PlotId, ConfigArc, PlotOpt}, frame::Data, data_artist_option_struct};

use super::{ArtistStyle, Artist, PathStyle, StyleCycle};

pub struct Container<M: Coord> {
    artists: Vec<ArtistStyle<M>>,
    style: PathStyle,
    cycle: StyleCycle,
}

impl<M: Coord> Container<M> {
    pub fn new() -> Self {
        Self {
            artists: Vec::new(),
            style: PathStyle::new(),
            cycle: StyleCycle::new(),
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
        clip: &Clip,
        style: &dyn PathOpt,
    ) {
        //let style_cycle = Style::new();
        //let style = Style::chain(style, &style_cycle);
        let style = self.style.push(style);

        for (i, artist) in self.artists.iter_mut().enumerate() {
            let style = self.cycle.push(&style, i);

            artist.draw(renderer, to_device, clip, &style);
        }
    }
}

impl PlotArtist<Data> for Container<Data> {
    type Opt = ContainerOpt;

    fn config(&mut self, cfg: &ConfigArc, id: PlotId) -> Self::Opt {
        self.cycle = StyleCycle::from_config(cfg, "container.cycle");

        unsafe { ContainerOpt::new(id) }
    }
}

data_artist_option_struct!(ContainerOpt, Container<Data>);

//impl PathStyleArtist for Container<Data> {
//    fn style_mut(&mut self) -> &mut PathStyle {
//        &mut self.style
//    }
//}
