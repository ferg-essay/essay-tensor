use core::fmt;

use essay_tensor::Tensor;

use crate::{driver::{Renderer, Device}, plot::PlotOpt, 
    artist::{Lines2d, ArtistTrait, Collection, Artist, patch, Angle, Container, Bezier3, Bezier2}, 
    figure::Point
};

use super::{rect::Rect, Bounds, Data, Affine2d, Figure};

pub struct Axes {
    pos_figure: Bounds<Figure>, // position of the Axes in figure grid coordinates
    pos_device: Bounds<Device>,

    data_lim: Bounds<Data>, // rectangle in data coordinates
    view_lim: Bounds<Data>, // rectangle in data coordinates

    to_device: Affine2d,

    artists: Vec<Box<dyn ArtistTrait>>,
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

    pub fn pie(
        &mut self, 
        x: impl Into<Tensor>, 
        opt: impl Into<PlotOpt>
    ) { // -> &mut Artist {
        let x = x.into();
        
        assert!(x.rank() == 1, "pie chart must have rank 1 data");

        let sum = x.reduce_sum()[0];

        let x = x / sum;

        let radius = 1.;

        let startangle = 0.;
        let mut theta1 = startangle / 360.;
        let center = Point(0., 0.);

        let mut container = Container::new();
        for frac in x.iter() {
            let theta2 = (theta1 - frac + 1.) % 1.;
            let patch = patch::Wedge::new(
                center, 
                radius, 
                Angle(theta1, theta2)
            );
            
            container.push(Artist::new(patch));

            theta1 = theta2;
        }

        //self.data_lim = Bounds::<Data>::new(
        //    Point(-1.25 + center.x(), 1.25 + center.y()))

        self.artist(container);

        // todo!()
    }

    pub fn plot(
        &mut self, 
        x: impl Into<Tensor>, 
        y: impl Into<Tensor>, 
        opt: impl Into<PlotOpt>
    ) -> &Box<dyn ArtistTrait> {
        let lines = Lines2d::from_xy(x, y);

        self.artist(lines)
    }

    pub fn bezier(
        &mut self, 
        p0: impl Into<Point>,
        p1: impl Into<Point>,
        p2: impl Into<Point>,
        p3: impl Into<Point>
    ) {
        self.artist(Bezier3(p0.into(), p1.into(), p2.into(), p3.into()));
    }

    pub fn bezier2(
        &mut self, 
        p0: impl Into<Point>,
        p1: impl Into<Point>,
        p2: impl Into<Point>,
    ) {
        self.artist(Bezier2(p0.into(), p1.into(), p2.into()));
    }

    pub fn scatter(
        &mut self, 
        x: impl Into<Tensor>, 
        y: impl Into<Tensor>, 
        opt: impl Into<PlotOpt>
    ) -> &Box<dyn ArtistTrait> {
        let collection = Collection::from_xy(x, y);

        self.artist(collection)
    }

    pub fn artist(&mut self, artist: impl ArtistTrait + 'static) -> &mut Box<dyn ArtistTrait> {
        let mut artist = artist;
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