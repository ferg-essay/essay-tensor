use core::fmt;

use essay_plot_base::{
    driver::{Renderer}, Style, StyleOpt,
    Bounds, Affine2d, Point, Canvas, CoordMarker,
};

use crate::artist::{Artist, ArtistTrait};

pub struct DataBox {
    pos_canvas: Bounds<Canvas>,

    data_bounds: Bounds<Data>,
    view_bounds: Bounds<Data>,

    artists: Vec<Artist<Data>>,

    to_canvas: Affine2d,
    style: Style,

}

impl DataBox {
    pub fn new() -> Self {
        Self {
            pos_canvas: Bounds::none(),

            data_bounds: Bounds::<Data>::unit(),
            view_bounds: Bounds::<Data>::unit(),

            artists: Vec::new(),

            style: Style::default(),

            to_canvas: Affine2d::eye(),
        }
    }

    ///
    /// Sets the canvas bounds
    /// 
    pub(crate) fn set_pos(&mut self, pos: &Bounds<Canvas>) -> &mut Self {
        self.pos_canvas = pos.clone();

        self.to_canvas = self.view_bounds.affine_to(&self.pos_canvas);

        self
    }

    pub fn artist(&mut self, artist: impl ArtistTrait<Data> + 'static) -> &mut Artist<Data> {
        let mut artist = artist;

        let bounds = artist.get_bounds();

        self.add_data_bounds(&bounds);
        self.add_view_bounds(&bounds);

        self.artists.push(Artist::new(artist));

        let len = self.artists.len();
        &mut self.artists[len - 1]
    }

    fn add_data_bounds(&mut self, bounds: &Bounds<Data>) {
        if self.artists.len() == 0 {
            self.data_bounds = bounds.clone();
        } else {
            self.data_bounds = self.data_bounds.union(&bounds);
        }
    }

    fn add_view_bounds(&mut self, bounds: &Bounds<Data>) {
        let x_margin = 0.1;
        let y_margin = 0.1;

        let (height, width) = (bounds.height(), bounds.width());

        let (mut xmin, mut xmax) = (bounds.xmin(), bounds.xmax());
        xmin -= x_margin * width;
        xmax += x_margin * width;

        let (mut ymin, mut ymax) = (bounds.ymin(), bounds.ymax());
        ymin -= y_margin * height;
        ymax += y_margin * height;

        let bounds = Bounds::new(Point(xmin, ymin), Point(xmax, ymax));

        if self.artists.len() == 0 {
            self.view_bounds = bounds;
        } else {
            self.view_bounds = self.view_bounds.union(&bounds);
        }
    }

    pub(crate) fn get_pos(&self) -> &Bounds<Canvas> {
        &self.pos_canvas
    }

    pub(crate) fn get_view_bounds(&self) -> &Bounds<Data> {
        &self.view_bounds
    }
}

impl ArtistTrait<Canvas> for DataBox {
    fn draw(
        &mut self, 
        renderer: &mut dyn Renderer, 
        _to_canvas: &Affine2d,
        _clip: &Bounds<Canvas>,
        style: &dyn StyleOpt,
    ) {
        //let to_canvas = to_canvas.matmul(&self.to_canvas);
        let to_canvas = &self.to_canvas;
        let style = Style::chain(style, &self.style);

        // TODO: intersect clip
        for artist in &mut self.artists {
            artist.draw(renderer, &to_canvas, &self.pos_canvas, &style);
        }
    }

    fn get_bounds(&mut self) -> Bounds<Canvas> {
        self.pos_canvas.clone()
    }
}


impl fmt::Debug for DataBox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DataBox({},{},{}x{})",
            self.view_bounds.xmin(),
            self.view_bounds.ymin(),
            self.view_bounds.width(),
            self.view_bounds.height())
    }
}

///
/// Data coordinates
///
#[derive(Clone, Copy, Debug)]
pub struct Data;

impl CoordMarker for Data {
}
