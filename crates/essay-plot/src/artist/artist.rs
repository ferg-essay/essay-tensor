use crate::{driver::{Renderer, Device}, figure::{Bounds, Data, Affine2d}};

pub struct Artist {
    artist: Box<dyn ArtistTrait>,
}

impl Artist {
    pub fn new(artist: impl ArtistTrait + 'static) -> Self {
        Self {
            artist: Box::new(artist),
        }
    }

    pub(crate) fn get_data_bounds(&mut self) -> Bounds<Data> {
        self.artist.get_data_bounds()
    }

    pub(crate) fn draw(
        &mut self, 
        renderer: &mut dyn Renderer, 
        to_device: &Affine2d, 
        clip: &Bounds<Device>
    ) {
        self.artist.draw(renderer, to_device, clip);
    }
}

pub trait ArtistTrait {
    fn get_data_bounds(&mut self) -> Bounds<Data>;
    
    fn draw(
        &mut self, 
        renderer: &mut dyn Renderer,
        to_device: &Affine2d,
        clip: &Bounds<Device>,
    );
}

