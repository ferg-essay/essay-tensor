use core::fmt;
use std::f32::consts::PI;

use essay_tensor::Tensor;

use crate::{driver::{Renderer, Canvas}, plot::PlotOpt, 
    artist::{Lines2d, ArtistTrait, Collection, Artist, patch, Angle, Container, Bezier3, Bezier2, ColorCycle, Style, Color, Text}, figure::GridSpec, prelude::Figure, frame::Point, 
};

use super::{Bounds, Data, Affine2d, databox::DataBox, frame::{Frame, BottomFrame, LeftFrame, RightFrame, TopFrame}};

pub struct Axes {
    pos_grid: Bounds<GridSpec>, // position of the Axes in figure grid coordinates
    pos: Bounds<Canvas>,

    title: Text,
    frame: Frame,

    to_canvas: Affine2d,
    style: Style,
}

impl Axes {
    pub fn new(bounds: impl Into<Bounds<GridSpec>>) -> Self {
        let mut axes = Self {
            pos_grid: bounds.into(),
            pos: Bounds::none(),

            title: Text::new(),
            frame: Frame::new(),

            style: Style::default(),

            to_canvas: Affine2d::eye(),
        };

        axes.default_properties();

        axes
    }

    fn default_properties(&mut self) {
        self.title.font().size(24.);
    }

    ///
    /// Sets the device bounds and propagates to children
    /// 
    pub(crate) fn set_pos(&mut self, pos: &Bounds<Canvas>) {
        self.pos = pos.clone();

        let title_bounds = self.title.get_bounds();

        let title_pos = Bounds::new(
            Point(pos.xmid(), pos.ymax() - title_bounds.height()),
            Point(pos.xmid(), pos.ymax())
        );

        self.title.set_pos(title_pos);

        let frame_pos = Bounds::new(
            Point(pos.xmin(), pos.ymin()),
            Point(pos.xmax(), pos.ymax() - title_bounds.height()),
        );

        self.frame.set_pos(&frame_pos);
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
        let colors = ColorCycle::tableau();
        let mut i = 0;
        for frac in x.iter() {
            let theta2 = (theta1 - frac + 1.) % 1.;
            let patch = patch::Wedge::new(
                center, 
                radius, 
                Angle(theta1, theta2)
            );

            //patch.color(colors[i]);
            let mut artist = Artist::new(patch);
            artist.color(colors[i]);
            
            container.push(artist);

            theta1 = theta2;
            i += 1;
        }

        self.frame.data_mut().artist(container);

        // todo!()
    }

    pub fn plot(
        &mut self, 
        x: impl Into<Tensor>, 
        y: impl Into<Tensor>, 
        opt: impl Into<PlotOpt>
    ) -> &mut Artist<Data> {
        let lines = Lines2d::from_xy(x, y);

        //self.artist(lines)
        self.frame.data_mut().artist(lines)
    }

    pub fn bezier3(
        &mut self, 
        p0: impl Into<Point>,
        p1: impl Into<Point>,
        p2: impl Into<Point>,
        p3: impl Into<Point>
    ) {
        self.frame.data_mut().artist(Bezier3(p0.into(), p1.into(), p2.into(), p3.into()));
    }

    pub fn bezier2(
        &mut self, 
        p0: impl Into<Point>,
        p1: impl Into<Point>,
        p2: impl Into<Point>,
    ) {
        self.frame.data_mut().artist(Bezier2(p0.into(), p1.into(), p2.into()));
    }

    pub fn scatter(
        &mut self, 
        x: impl Into<Tensor>, 
        y: impl Into<Tensor>, 
        opt: impl Into<PlotOpt>
    ) -> &Artist<Data> {
        let collection = Collection::from_xy(x, y);

        self.frame.data_mut().artist(collection)
    }

    pub(crate) fn draw(&mut self, renderer: &mut impl Renderer) {
        self.title.draw(renderer, &self.to_canvas, &self.pos, &self.style);

        self.frame.draw(renderer);
    }

    pub fn title(&mut self, text: &str) -> &mut Text {
        self.title.text(text);

        &mut self.title
    }

    pub fn xlabel(&mut self, text: &str) -> &mut Text {
        self.frame.xlabel(text)
    }

    pub fn ylabel(&mut self, text: &str) -> &mut Text {
        self.frame.ylabel(text)
    }
}

impl fmt::Debug for Axes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Axes({},{},{}x{})",
            self.frame.pos().xmin(),
            self.frame.pos().ymin(),
            self.frame.pos().width(),
            self.frame.pos().height())
    }
}

impl From<()> for Axes {
    fn from(_value: ()) -> Self {
        Axes::new(Bounds::<GridSpec>::none())
    }
}