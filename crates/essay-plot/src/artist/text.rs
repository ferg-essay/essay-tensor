use crate::{frame::{Bounds, Point}, driver::Canvas};

use super::{Style, ArtistTrait, style::Chain};

pub struct Text {
    pos: Bounds<Canvas>,

    text: Option<String>,

    style: Style,
    text_style: TextStyle,

    angle: f32,
}

impl Text {
    pub fn new() -> Self {
        Self {
            pos: Bounds::none(),
            text: None,

            style: Style::new(),
            text_style: TextStyle::new(),

            angle: 0.
        }
    }

    pub(crate) fn set_pos(&mut self, pos: Bounds<Canvas>) {
        self.pos = pos;
    }

    pub(crate) fn get_pos(&self) -> &Bounds<Canvas> {
        &self.pos
    }

    pub fn text(&mut self, text: &str) -> &mut Self {
        if text.len() > 0 {
            self.text = Some(text.to_string());
        } else {
            self.text = None;
        }

        self
    }

    pub fn font(&mut self) -> &mut TextStyle {
        &mut self.text_style
    }

    pub fn style(&mut self) -> &mut Style {
        &mut self.style
    }

    pub fn angle(&mut self, angle: f32) -> &mut Self {
        self.angle = angle;

        self
    }

    pub fn get_angle(&self) -> f32 {
        self.angle
    }
}

impl ArtistTrait<Canvas> for Text {
    fn get_bounds(&mut self) -> Bounds<Canvas> {
        match &self.text {
            None => Bounds::zero(),
            Some(text) => {
                let size = match self.text_style.get_size() {
                    Some(size) => *size,
                    None => 16.,
                };

                let width = text.len() as f32 * size as f32 * 0.5;

                Bounds::extent(width, 2. * size)
            }
        }
    }

    fn draw(
        &mut self, 
        renderer: &mut dyn crate::driver::Renderer,
        to_canvas: &crate::frame::Affine2d,
        clip: &Bounds<Canvas>,
        style: &dyn super::StyleOpt,
    ) {
        if let Some(text) = &self.text {
            let style = Chain::new(style, &self.style);

            if ! self.pos.is_none() {
                renderer.draw_text(
                    Point(self.pos.xmid(), self.pos.ymid()),
                    text,
                    self.get_angle(),
                    &style,
                    &self.text_style,
                    clip
                ).unwrap();
            }
        }
    }
}

pub struct TextStyle {
    size: Option<f32>,
}

impl TextStyle {
    pub fn new() -> Self {
        Self {
            size: None,
        }
    }

    #[inline]
    pub fn get_size(&self) -> &Option<f32> {
        &self.size
    }

    pub fn size(&mut self, size: f32) -> &mut Self {
        self.size = Some(size);

        self
    }
}