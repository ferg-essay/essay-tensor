use essay_plot_base::{
    Affine2d, 
    Bounds, Point, Canvas,
    PathOpt,
    driver::Renderer, 
    TextStyle,
};

use super::{Artist, PathStyle};

pub struct Text {
    pos: Bounds<Canvas>,
    extent: Bounds<Canvas>,

    text: Option<String>,

    style: PathStyle,
    text_style: TextStyle,

    angle: f32,
}

impl Text {
    pub const DESC : f32 = 0.3;

    pub fn new() -> Self {
        Self {
            pos: Bounds::none(),
            extent: Bounds::zero(),
            text: None,

            style: PathStyle::new(),
            text_style: TextStyle::new(),

            angle: 0.
        }
    }

    pub(crate) fn set_pos(&mut self, pos: impl Into<Bounds<Canvas>>) {
        self.pos = pos.into();
    }

    pub fn text(&mut self, text: &str) -> &mut Self {
        if text.len() > 0 {
            self.text = Some(text.to_string());
        } else {
            self.text = None;
        }

        self
    }

    pub fn height(&self) -> f32 {
        2. * 14.
    }

    pub fn font(&mut self) -> &mut TextStyle {
        &mut self.text_style
    }

    pub fn style(&mut self) -> &mut PathStyle {
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

impl Artist<Canvas> for Text {
    fn get_extent(&mut self) -> Bounds<Canvas> {
        self.extent.clone()
    }

    fn update(&mut self, canvas: &Canvas) {
        self.extent = match &self.text {
            None => Bounds::zero(),
            Some(text) => {
                let size = match self.text_style.get_size() {
                    Some(size) => *size,
                    None => TextStyle::SIZE_DEFAULT,
                };

                let width = text.len() as f32 * size as f32 * 0.5;

                Bounds::extent(
                    width * canvas.scale_factor(), 
                    size * (1. + Self::DESC) * canvas.scale_factor()
                )
            }
        }
    }

    fn draw(
        &mut self, 
        renderer: &mut dyn Renderer,
        _to_canvas: &Affine2d,
        clip: &Bounds<Canvas>,
        style: &dyn PathOpt,
    ) {
        if let Some(text) = &self.text {
            let style = self.style.push(style);

            if ! self.pos.is_none() {
                let desc = Self::DESC * self.extent.height();

                renderer.draw_text(
                    Point(self.pos.xmid(), self.pos.ymin() + desc),
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
