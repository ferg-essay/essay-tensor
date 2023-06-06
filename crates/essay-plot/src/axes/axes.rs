use core::fmt;

use essay_tensor::Tensor;

use crate::{device::{Renderer, Device}, figure::Figure, plot::PlotOpt, artist::{Lines2d, Artist, Collection}};

use super::{rect::Rect, Bounds, Data, Affine2d};

pub struct Axes {
    pos_figure: Bounds<Figure>, // position of the Axes in figure grid coordinates
    pos_device: Bounds<Device>,

    data_lim: Bounds<Data>, // rectangle in data coordinates
    view_lim: Bounds<Data>, // rectangle in data coordinates

    to_device: Affine2d,

    artists: Vec<Box<dyn Artist>>,
}

impl Axes {
    pub fn new(bounds: impl Into<Bounds<Figure>>) -> Self {
        Self {
            pos_figure: bounds.into(),
            pos_device: Bounds::none(),

            artists: Vec::new(),

            view_lim: Bounds::<Data>::unit(),
            data_lim: Bounds::<Data>::unit(),

            to_device: Affine2d::eye(),

        }
    }

    ///
    /// Sets the device bounds and propagates to children
    /// 
    pub(crate) fn bounds(&mut self, pos: &Bounds<Device>) -> &mut Self {
        self.pos_device = pos.clone();

        self.to_device = self.view_lim.affine_to(&self.pos_device);

        self
    }

    pub fn plot(
        &mut self, 
        x: impl Into<Tensor>, 
        y: impl Into<Tensor>, 
        opt: impl Into<PlotOpt>
    ) -> &Box<dyn Artist> {
        let lines = Lines2d::from_xy(x, y);

        self.artist(lines)
    }

    pub fn scatter(
        &mut self, 
        x: impl Into<Tensor>, 
        y: impl Into<Tensor>, 
        opt: impl Into<PlotOpt>
    ) -> &Box<dyn Artist> {
        let collection = Collection::from_xy(x, y);

        self.artist(collection)
    }

    pub fn artist(&mut self, artist: impl Artist + 'static) -> &mut Box<dyn Artist> {
        let bounds = artist.get_data_bounds();

        if self.artists.len() == 0 {
            self.data_lim = bounds;
            self.view_lim = self.data_lim.clone();
        } else {
            self.data_lim = self.data_lim.union(bounds);
            self.view_lim = self.data_lim.clone();
        }

        self.artists.push(Box::new(artist));

        let len = self.artists.len();
        &mut self.artists[len - 1]
    }

    pub(crate) fn draw(&mut self, renderer: &mut dyn Renderer) {
        for artist in &mut self.artists {
            artist.draw(renderer, &self.to_device, &self.pos_device);
        }
    }

    /*
    /// only includes data extent, not labels or axes
    pub fn get_window_extent(&self, _renderer: Option<&dyn Renderer>) -> &Bounds {
        &self.position
    }
    */
}

impl fmt::Debug for Axes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Axes({},{},{}x{})",
            self.view_lim.xmin(),
            self.view_lim.ymin(),
            self.view_lim.width(),
            self.view_lim.height())
    }
}

impl From<()> for Axes {
    fn from(_value: ()) -> Self {
        Axes::new(Bounds::<Figure>::none())
    }
}