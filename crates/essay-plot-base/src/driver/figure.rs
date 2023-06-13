use super::{Renderer};
use crate::{CanvasEvent, Bounds, Canvas};

pub trait FigureApi {
    fn draw(&mut self, renderer: &mut dyn Renderer, bounds: &Bounds<Canvas>);

    fn event(&mut self, renderer: &mut dyn Renderer, event: &CanvasEvent);
}