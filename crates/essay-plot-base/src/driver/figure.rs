use super::Renderer;

pub trait FigureApi {
    fn draw(&mut self, renderer: &mut dyn Renderer);
}