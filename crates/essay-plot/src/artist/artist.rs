use crate::{driver::{Renderer, Device}, figure::{Bounds, Data, Affine2d}};

use super::{Style, StyleOpt, Color, JoinStyle};

pub struct Artist {
    artist: Box<dyn ArtistTrait>,

    style: Style,
}

impl Artist {
    pub fn new(artist: impl ArtistTrait + 'static) -> Self {
        Self {
            artist: Box::new(artist),
            style: Style::default(),
        }
    }
}

impl ArtistTrait for Artist {
    fn get_data_bounds(&mut self) -> Bounds<Data> {
        self.artist.get_data_bounds()
    }

    fn draw(
        &mut self, 
        renderer: &mut dyn Renderer, 
        to_device: &Affine2d, 
        clip: &Bounds<Device>,
        style: &dyn StyleOpt,
    ) {
        self.artist.draw(renderer, to_device, clip, &Style::chain(style, &self.style));
    }
}

pub trait ArtistTrait {
    fn get_data_bounds(&mut self) -> Bounds<Data>;
    
    fn draw(
        &mut self, 
        renderer: &mut dyn Renderer,
        to_device: &Affine2d,
        clip: &Bounds<Device>,
        style: &dyn StyleOpt,
    );
}

impl StyleOpt for Artist {
    fn get_color(&self) -> &Option<super::Color> {
        self.style.get_color()
    }

    fn get_linewidth(&self) -> &Option<f32> {
        self.style.get_linewidth()
    }

    fn get_joinstyle(&self) -> &Option<JoinStyle> {
        self.style.get_joinstyle()
    }
}

impl Artist {
    pub fn color(&mut self, color: impl Into<Color>) -> &mut Self {
        self.style.color(color);

        self
    }

    pub fn linewidth(&mut self, width: f32) -> &mut Self {
        self.style.linewidth(width);

        self
    }

    pub fn joinstyle(&mut self, joinstyle: impl Into<JoinStyle>) -> &mut Self {
        self.style.joinstyle(joinstyle);

        self
    }
}
