use essay_plot_base::{
    CoordMarker, Bounds, Affine2d, Canvas, StyleOpt, Style,
    driver::Renderer, JoinStyle, Color,
};

use crate::frame::ArtistId;

pub trait ArtistTrait<M: CoordMarker> {
    fn update_extent(&mut self, canvas: &Canvas) {}

    fn get_extent(&mut self) -> Bounds<M>;
    
    fn draw(
        &mut self, 
        renderer: &mut dyn Renderer,
        to_canvas: &Affine2d,
        clip: &Bounds<Canvas>,
        style: &dyn StyleOpt,
    );
}

pub struct Artist<M: CoordMarker> {
    id: ArtistId,

    artist: Box<dyn ArtistTrait<M>>,

    style: Style,
}

impl<M: CoordMarker> Artist<M> {
    pub fn new(id: ArtistId, artist: impl ArtistTrait<M> + 'static) -> Self {
        Self {
            id,
            artist: Box::new(artist),
            style: Style::default(),
        }
    }

    pub fn id(&self) -> ArtistId {
        self.id
    }

    pub fn style_mut(&mut self) -> &mut Style {
        &mut self.style
    }
}

impl<M: CoordMarker> ArtistTrait<M> for Artist<M> {
    fn update_extent(&mut self, canvas: &Canvas) {
        self.artist.update_extent(canvas)
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

impl<M: CoordMarker> StyleOpt for Artist<M> {
    fn get_color(&self) -> &Option<Color> {
        self.style.get_color()
    }

    fn get_linewidth(&self) -> &Option<f32> {
        self.style.get_linewidth()
    }

    fn get_joinstyle(&self) -> &Option<JoinStyle> {
        self.style.get_joinstyle()
    }
}

impl<M: CoordMarker> Artist<M> {
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
