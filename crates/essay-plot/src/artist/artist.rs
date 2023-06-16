use essay_plot_base::{
    Coord, Bounds, Affine2d, Canvas, StyleOpt, Style,
    driver::Renderer, JoinStyle, Color, CapStyle,
};

use crate::frame::ArtistId;

pub trait Artist<M: Coord> {
    fn update(&mut self, canvas: &Canvas);

    fn get_extent(&mut self) -> Bounds<M>;
    
    fn draw(
        &mut self, 
        renderer: &mut dyn Renderer,
        to_canvas: &Affine2d,
        clip: &Bounds<Canvas>,
        style: &dyn StyleOpt,
    );
}

pub struct ArtistStyle<M: Coord> {
    id: ArtistId,

    artist: Box<dyn Artist<M>>,

    style: Style,
}

impl<M: Coord> ArtistStyle<M> {
    pub fn new(id: ArtistId, artist: impl Artist<M> + 'static) -> Self {
        Self {
            id,
            artist: Box::new(artist),
            style: Style::default(),
        }
    }

    pub fn id(&self) -> ArtistId {
        self.id
    }

    pub fn style(&self) -> &Style {
        &self.style
    }

    pub fn style_mut(&mut self) -> &mut Style {
        &mut self.style
    }
    
    pub fn color(&mut self, color: impl Into<Color>) -> &mut Self {
        self.style.color(color);

        self
    }

    pub fn linewidth(&mut self, width: f32) -> &mut Self {
        self.style.linewidth(width);

        self
    }

    pub fn joinstyle(&mut self, joinstyle: impl Into<JoinStyle>) -> &mut Self {
        self.style.joinstyle(joinstyle);

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
        style: &dyn StyleOpt,
    ) {
        self.artist.draw(
            renderer, 
            to_canvas,
            clip,
            &Style::chain(style, &self.style)
        );
    }
}

impl<M: Coord> StyleOpt for ArtistStyle<M> {
    fn get_facecolor(&self) -> &Option<Color> {
        self.style.get_facecolor()
    }

    fn get_edgecolor(&self) -> &Option<Color> {
        self.style.get_edgecolor()
    }

    fn get_linewidth(&self) -> &Option<f32> {
        self.style.get_linewidth()
    }

    fn get_joinstyle(&self) -> &Option<JoinStyle> {
        self.style.get_joinstyle()
    }

    fn get_capstyle(&self) -> &Option<CapStyle> {
        self.style.get_capstyle()
    }
}
