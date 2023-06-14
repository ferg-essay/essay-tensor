use core::fmt;

use essay_tensor::Tensor;

use essay_plot_base::{
    driver::{Renderer}, 
    Point, Bounds, Canvas, Affine2d, Style, Angle, Path, CanvasEvent,
};

use crate::{artist::{
    Lines2d, ArtistTrait, Collection, Artist, patch, Container, Bezier3, Bezier2, ColorCycle, Text, paths, PColor, 
}, prelude::PlotOpt};

use crate::graph::{Layout, GraphId};

use super::{Data, frame::{Frame}};

pub struct Graph {
    id: GraphId,

    pos: Bounds<Canvas>,

    pos_layout: Bounds<Layout>, // position of the Graph in layout coordinates

    title: Text,
    frame: Frame,

    to_canvas: Affine2d,
    style: Style,
}

impl Graph {
    pub(crate) fn new(id: GraphId, layout: impl Into<Bounds<Layout>>) -> Self {
        let mut graph = Self {
            id, 
            pos_layout: layout.into(),
            pos: Bounds::none(),

            title: Text::new(),
            frame: Frame::new(),

            style: Style::default(),

            to_canvas: Affine2d::eye(),
        };

        graph.default_properties();

        graph
    }

    #[inline]
    pub fn id(&self) -> GraphId {
        self.id
    }

    pub fn pos(&self) -> &Bounds<Canvas> {
        &self.pos
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

    fn default_properties(&mut self) {
        self.title.font().size(12.);
    }

    ///
    /// Calculate the graph's extents
    /// 
    pub(crate) fn extent(&mut self, canvas: &Canvas) {
        self.title.update_extent(canvas);
        self.frame.update_extent(canvas);
    }

    ///
    /// Sets the device bounds and propagates to children
    /// 
    pub(crate) fn set_pos(&mut self, pos: &Bounds<Canvas>) {
        self.pos = pos.clone();

        let title_bounds = self.title.get_extent();

        let margin = 10.;
        let title_gap = 10.;

        let title_pos = Bounds::new(
            Point(pos.xmid(), pos.ymax() - margin - title_bounds.height()),
            Point(pos.xmid(), pos.ymax() - margin)
        );

        self.title.set_pos(title_pos);

        let frame_pos = Bounds::new(
            Point(pos.xmin(), pos.ymin()),
            Point(pos.xmax(), pos.ymax() - margin - title_bounds.height() - title_gap),
        );

        self.frame.set_pos(&frame_pos);
    }

    pub(crate) fn event(&mut self, renderer: &mut dyn Renderer, event: &CanvasEvent) {
        self.frame.event(renderer, event);
    }

    //
    // TODO: move plots out of graph
    //

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
                (Angle::Unit(theta1), Angle::Unit(theta2))
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

    pub fn pcolor(
        &mut self, 
    ) {
        self.frame.data_mut().artist(PColor {});
    }

    pub fn scatter(
        &mut self, 
        x: impl Into<Tensor>, 
        y: impl Into<Tensor>, 
        _opt: impl Into<PlotOpt>
    ) -> &mut Artist<Data> {
        let x : Tensor = x.into();
        let y = y.into();
        let xy = x.stack(&[y], -1); // Axis::axis(-1));
        let path: Path<Canvas> = paths::unit().transform(&Affine2d::eye().scale(10., 10.));

        let collection = Collection::new(path, &xy);

        self.frame.data_mut().artist(collection)
    }

    pub(crate) fn draw(&mut self, renderer: &mut dyn Renderer) {
        self.title.draw(renderer, &self.to_canvas, &self.pos, &self.style);

        self.frame.draw(renderer);
    }
}

impl fmt::Debug for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Graph({},{}; {}x{})",
            self.frame.pos().xmin(),
            self.frame.pos().ymin(),
            self.frame.pos().width(),
            self.frame.pos().height())
    }
}
