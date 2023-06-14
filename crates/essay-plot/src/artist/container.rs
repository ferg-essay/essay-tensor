use essay_plot_base::{
    Bounds, Affine2d, Point, CoordMarker, Canvas, StyleOpt, Style,
    driver::{Renderer}
};

use super::{Artist, ArtistTrait};

pub struct Container<M: CoordMarker> {
    artists: Vec<Artist<M>>,
    style: Style,
}

impl<M: CoordMarker> Container<M> {
    pub fn new() -> Self {
        Self {
            artists: Vec::new(),
            style: Style::new(),
        }
    }

    fn style_mut(&mut self) -> &mut Style {
        &mut self.style
    }

    pub fn push(&mut self, artist: Artist<M>) {
        self.artists.push(artist);
    }
}

impl<M: CoordMarker> ArtistTrait<M> for Container<M> {
    fn update_extent(&mut self, canvas: &Canvas) {
        for artist in &mut self.artists {
            artist.update_extent(canvas);
        }
    }

    fn get_extent(&mut self) -> Bounds<M> {
        let mut bounds : Option<Bounds<M>> = None;

        for artist in &mut self.artists {
            bounds = match bounds {
                None => Some(artist.get_extent().clone()),
                Some(bounds) => Some(bounds.union(&artist.get_extent())),
            }
        }

        bounds.unwrap()
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
