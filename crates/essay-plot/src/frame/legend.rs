use essay_plot_base::{Bounds, Canvas, Clip};

use crate::artist::Artist;

pub struct Legend {
    pos: Bounds<Canvas>,
    extent: Bounds<Canvas>,
}

impl Legend {
    pub fn new() -> Self {
        Self {
            pos: Bounds::zero(),
            extent: Bounds::zero(),
        }
    }

    pub fn set_pos(&mut self, pos: Bounds<Canvas>) {
        self.pos = pos;
    }
}

impl Artist<Canvas> for Legend {
    fn update(&mut self, _canvas: &Canvas) {
    }

    fn get_extent(&mut self) -> Bounds<Canvas> {
        self.extent.clone()
    }

    fn draw(
        &mut self, 
        renderer: &mut dyn essay_plot_base::driver::Renderer,
        to_canvas: &essay_plot_base::Affine2d,
        clip: &Clip,
        style: &dyn essay_plot_base::PathOpt,
    ) {

    }
}