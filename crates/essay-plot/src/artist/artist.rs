use crate::{driver::{Renderer, Device}, figure::{Bounds, Data, Affine2d}};

pub trait Artist {
    fn get_data_bounds(&self) -> Bounds<Data>;
    
    fn draw(
        &mut self, 
        renderer: &mut dyn Renderer,
        to_device: &Affine2d,
        clip: &Bounds<Device>,
    );
}