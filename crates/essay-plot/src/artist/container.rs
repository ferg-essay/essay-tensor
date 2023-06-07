use crate::{figure::{Bounds, Data, Affine2d, Point}, driver::{Renderer, Device}};

use super::{Artist, ArtistTrait, StyleOpt, Style};

pub struct Container {
    artists: Vec<Artist>,
}

impl Container {
    pub fn new() -> Self {
        Self {
            artists: Vec::new(),
        }
    }

    pub fn push(&mut self, artist: Artist) {
        self.artists.push(artist);
    }
}

impl ArtistTrait for Container {
    fn get_data_bounds(&mut self) -> Bounds<Data> {
        let mut bounds : Option<Bounds<Data>> = None;

        for artist in &mut self.artists {
            bounds = match bounds {
                None => Some(artist.get_data_bounds()),
                Some(bounds) => Some(bounds.union(artist.get_data_bounds())),
            }
        }

        bounds.unwrap();

        Bounds::<Data>::new(Point(-1.5, -1.5), Point(1.5, 1.5))
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
