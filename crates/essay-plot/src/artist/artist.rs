use crate::device::Renderer;

pub trait Artist {
    fn draw(&mut self, renderer: &mut dyn Renderer);
}