use essay_plot_base::{
    Coord, Bounds, Affine2d, Canvas, PathOpt,
    driver::Renderer, JoinStyle, Color, CapStyle, LineStyle,
};

use crate::frame::ArtistId;

use super::PathStyle;

pub trait Artist<M: Coord> {
    fn update(&mut self, canvas: &Canvas);

    fn get_extent(&mut self) -> Bounds<M>;
    
    fn draw(
        &mut self, 
        renderer: &mut dyn Renderer,
        to_canvas: &Affine2d,
        clip: &Bounds<Canvas>,
        style: &dyn PathOpt,
    );
}

pub struct ArtistStyle<M: Coord> {
    id: ArtistId,

    artist: Box<dyn Artist<M>>,

    style: PathStyle,
}

impl<M: Coord> ArtistStyle<M> {
    pub fn new(id: ArtistId, artist: impl Artist<M> + 'static) -> Self {
        Self {
            id,
            artist: Box::new(artist),
            style: PathStyle::default(),
        }
    }

    pub fn id(&self) -> ArtistId {
        self.id
    }

    pub fn style(&self) -> &PathStyle {
        &self.style
    }

    pub fn style_mut(&mut self) -> &mut PathStyle {
        &mut self.style
    }
    
    pub fn color(&mut self, color: impl Into<Color>) -> &mut Self {
        self.style.color(color);

        self
    }

    pub fn linewidth(&mut self, width: f32) -> &mut Self {
        self.style.line_width(width);

        self
    }

    pub fn join_style(&mut self, joinstyle: impl Into<JoinStyle>) -> &mut Self {
        self.style.join_style(joinstyle);

        self
    }
}

impl<M: Coord> Artist<M> for ArtistStyle<M> {
    fn update(&mut self, canvas: &Canvas) {
        self.artist.update(canvas)
    }

    fn get_extent(&mut self) -> Bounds<M> {
        self.artist.get_extent()
    }

    fn draw(
        &mut self, 
        renderer: &mut dyn Renderer, 
        to_canvas: &Affine2d, 
        clip: &Bounds<Canvas>,
        style: &dyn PathOpt,
    ) {
        self.artist.draw(
            renderer, 
            to_canvas,
            clip,
            //&Style::chain(style, &self.style)
            &self.style.push(style)
        );
    }
}

impl<M: Coord> PathOpt for ArtistStyle<M> {
    fn get_face_color(&self) -> &Option<Color> {
        self.style.get_face_color()
    }

    fn get_edge_color(&self) -> &Option<Color> {
        self.style.get_edge_color()
    }

    fn get_line_width(&self) -> &Option<f32> {
        self.style.get_line_width()
    }

    fn get_join_style(&self) -> &Option<JoinStyle> {
        self.style.get_join_style()
    }

    fn get_cap_style(&self) -> &Option<CapStyle> {
        self.style.get_cap_style()
    }

    fn get_line_style(&self) -> &Option<LineStyle> {
        self.style.get_line_style()
    }

    fn get_alpha(&self) -> &Option<f32> {
        todo!()
    }

    fn get_texture(&self) -> &Option<essay_plot_base::TextureId> {
        todo!()
    }
}
