use essay_plot_base::{Bounds, Canvas, Clip, PathOpt, Affine2d, driver::Renderer, PathCode, Path, Point};

use crate::artist::{Artist, PathStyle};

pub struct Legend {
    pos: Bounds<Canvas>,
    extent: Bounds<Canvas>,

    style: PathStyle,
}

impl Legend {
    pub fn new() -> Self {
        let mut legend = Self {
            pos: Bounds::zero(),
            extent: Bounds::zero(),
            style: PathStyle::new(),
        };

        legend.style.face_color("teal");

        legend
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
        renderer: &mut dyn Renderer,
        to_canvas: &Affine2d,
        clip: &Clip,
        style: &dyn PathOpt,
    ) {
        let pos = &self.pos;
        let w = 300.;
        let h = 150.;
        
        let path = Path::<Canvas>::new(vec![
            PathCode::MoveTo(Point(pos.x0(), pos.y0())),
            PathCode::LineTo(Point(pos.x0(), pos.y0() - h)),
            PathCode::LineTo(Point(pos.x0() + w, pos.y0() - h)),
            PathCode::ClosePoly(Point(pos.x0() + w, pos.y0())),
        ]);

        renderer.draw_path(&path, &self.style, clip).unwrap();
    }
}

pub trait LegendHandler {
    
}