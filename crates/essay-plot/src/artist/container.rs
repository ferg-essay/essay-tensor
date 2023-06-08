use crate::{axes::{Bounds, Data, Affine2d, Point, CoordMarker}, driver::{Renderer, Device}};

use super::{Artist, ArtistTrait, StyleOpt, Style};

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
    fn get_data_bounds(&mut self) -> Bounds<M> {
        let mut bounds : Option<Bounds<M>> = None;

        for artist in &mut self.artists {
            bounds = match bounds {
                None => Some(artist.get_data_bounds().clone()),
                Some(bounds) => Some(bounds.union(&artist.get_data_bounds())),
            }
        }

        bounds.unwrap();

        Bounds::new(Point(-1.5, -1.5), Point(1.5, 1.5))
    }

    fn draw(
        &mut self, 
        renderer: &mut dyn Renderer,
        to_device: &Affine2d,
        clip: &Bounds<Device>,
        style: &dyn StyleOpt,
    ) {
        //let style_cycle = Style::new();
        //let style = Style::chain(style, &style_cycle);

        for artist in &mut self.artists {
            artist.draw(renderer, to_device, clip, style);
        }
    }
}
