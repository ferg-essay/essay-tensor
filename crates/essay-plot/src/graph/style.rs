use essay_plot_base::{Coord, Canvas, Bounds, driver::Renderer, Affine2d, Clip, PathOpt};

use crate::{artist::{Artist, PathStyle}, data_artist_option_struct, frame::{Data, LayoutArc, ArtistId}, path_style_options};

use super::{PlotArtist, PlotId};

data_artist_option_struct!(PlotOpt2, PlotStyleArtist<Data>);

impl PlotOpt2 {
    path_style_options!(style);
}

pub struct PlotStyleArtist<M: Coord> {
    artist: Box<dyn Artist<M>>,
    style: PathStyle,
}

impl<M: Coord> PlotStyleArtist<M> {
    pub fn new<A>(artist: A) -> Self
    where
        A: Artist<M> + 'static
    {
        Self {
            artist: Box::new(artist),
            style: PathStyle::new(),
        }
    }
}

impl<M: Coord> Artist<M> for PlotStyleArtist<M> {
    fn update(&mut self, canvas: &Canvas) {
        self.artist.update(canvas);
    }

    fn get_extent(&mut self) -> Bounds<M> {
        self.artist.get_extent()
    }

    fn draw(
        &mut self, 
        renderer: &mut dyn Renderer,
        to_canvas: &Affine2d,
        clip: &Clip,
        style: &dyn PathOpt,
    ) {
        let style = self.style.push(style);

        self.artist.draw(renderer, to_canvas, clip, &style);
    }
}

impl<M: Coord> PlotArtist<M> for PlotStyleArtist<M> {
    type Opt = PlotOpt2;

    fn config(
        &mut self, 
        cfg: &super::ConfigArc, 
        id: PlotId,
    ) -> Self::Opt {
        self.style = PathStyle::from_config(cfg, "artist");

        PlotOpt2::new(id)
    }
}
