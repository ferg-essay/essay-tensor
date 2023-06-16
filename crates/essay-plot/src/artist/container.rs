use essay_plot_base::{
    Bounds, Affine2d, Point, Coord, Canvas, StyleOpt,
    driver::{Renderer}
};

use super::{ArtistStyle, Artist, Style};

pub struct Container<M: Coord> {
    artists: Vec<ArtistStyle<M>>,
    style: Style,
}

impl<M: Coord> Container<M> {
    pub fn new() -> Self {
        Self {
            artists: Vec::new(),
            style: Style::new(),
        }
    }

    fn style_mut(&mut self) -> &mut Style {
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
        style: &dyn StyleOpt,
    ) {
        //let style_cycle = Style::new();
        //let style = Style::chain(style, &style_cycle);

        for artist in &mut self.artists {
            artist.draw(renderer, to_device, clip, style);
        }
    }
}
