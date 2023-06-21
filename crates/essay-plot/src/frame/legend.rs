use essay_plot_base::{Bounds, Canvas, Clip, PathOpt, Affine2d, driver::Renderer, PathCode, Path, Point, TextStyle, VertAlign, HorizAlign};

use crate::artist::{Artist, PathStyle};

use super::data_box::DataBox;

pub struct Legend {
    pos: Bounds<Canvas>,
    extent: Bounds<Canvas>,

    handlers: Vec<LegendHandler>,

    path_style: PathStyle,
    text_style: TextStyle,

    glyph_size: f32,
}

impl Legend {
    pub fn new() -> Self {
        let mut legend = Self {
            pos: Bounds::zero(),
            extent: Bounds::zero(),

            path_style: PathStyle::new(),
            text_style: TextStyle::new(),

            handlers: Vec::new(),

            glyph_size: 0.,
        };

        legend.path_style.face_color("white").edge_color("#b0b0b0").line_width(1.);
        legend.text_style.valign(VertAlign::Top);
        legend.text_style.halign(HorizAlign::Left);

        legend
    }

    pub fn set_pos(&mut self, pos: Bounds<Canvas>) {
        self.pos = pos;
    }

    pub(crate) fn update_handlers(&mut self, data: &DataBox) {
        let handlers = data.get_handlers();
        self.handlers = handlers;
    }
}

impl Artist<Canvas> for Legend {
    fn update(&mut self, canvas: &Canvas) {
        let font_size = match self.text_style.get_size() {
            Some(size) => *size,
            None => 10.,
        };

        self.glyph_size = canvas.to_px(font_size);
    }

    fn get_extent(&mut self) -> Bounds<Canvas> {
        self.extent.clone()
    }

    fn draw(
        &mut self, 
        renderer: &mut dyn Renderer,
        _to_canvas: &Affine2d,
        clip: &Clip,
        style: &dyn PathOpt,
    ) {
        if self.handlers.len() == 0 {
            return;
        }

        let n_chars = match self.handlers.iter().map(|h| h.get_label().len()).max() {
            Some(n_chars) => n_chars,
            None => 0,
        };

        let dh = self.glyph_size;
        let label_width = self.glyph_size * 0.4 * n_chars as f32;

        let pos = &self.pos;
        let rect_width = 80.;
        //let rect_height = dh;
        let pad_x = 15.;
        let pad_y = 5.;
        let margin = 20.;

        let w = label_width + pad_x + rect_width + 2. * margin;
        let h = dh * self.handlers.len() as f32 + 2. * margin;

        let x0 = pos.x0() + margin;
        let y0 = pos.y0() - margin;

        let path = Path::<Canvas>::new(vec![
            PathCode::MoveTo(Point(x0, y0)),
            PathCode::LineTo(Point(x0, y0 - h)),
            PathCode::LineTo(Point(x0 + w, y0 - h)),
            PathCode::ClosePoly(Point(x0 + w, y0)),
        ]);

        renderer.draw_path(&path, &self.path_style, clip).unwrap();

        let x_symbol = x0 + margin;
        let x_t = x_symbol + rect_width + pad_x;

        for (i, handler) in self.handlers.iter().enumerate() {
            let y = y0 - i as f32 * (dh + pad_y) - margin;
            renderer.draw_text(
                Point(x_t, y),
                handler.get_label(),
                0.,
                style,
                &self.text_style,
                clip
            ).unwrap();

            let rect = Bounds::<Canvas>::new(
                Point(x_symbol, y), 
                Point(x_symbol + rect_width, y - dh)
             );

             handler.draw(renderer, &rect);
        }
    }
}

pub struct LegendHandler {
    label: String,
    draw: Box<dyn Fn(&mut dyn Renderer, &Bounds<Canvas>)>,
}

impl LegendHandler {
    pub fn new(
        label: String,
        draw: impl Fn(&mut dyn Renderer, &Bounds<Canvas>) + 'static
    ) -> Self {
        Self {
            label,
            draw: Box::new(draw),
        }
    }

    pub fn get_label(&self) -> &String {
        &self.label
    }

    pub fn draw(&self, renderer: &mut dyn Renderer, rect: &Bounds<Canvas>) {
        (self.draw)(renderer, rect);
    }
}