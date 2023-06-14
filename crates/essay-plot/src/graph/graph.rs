use core::fmt;

use essay_plot_base::{
    driver::{Renderer}, 
    Bounds, Canvas, CanvasEvent,
};

use crate::{artist::{
    ArtistTrait, Artist,
    Text,
}, frame::{Frame, Data}};

use crate::frame::{Layout};

use super::GraphId;

pub struct Graph {
    id: GraphId,

    pos: Bounds<Canvas>,

    frame: Frame,
}

impl Graph {
    pub(crate) fn new(id: GraphId, _layout: impl Into<Bounds<Layout>>) -> Self {
        let mut graph = Self {
            id, 
            //pos_layout: layout.into(),
            pos: Bounds::none(),

            //title: Text::new(),
            frame: Frame::new(),

            //style: Style::default(),

            //to_canvas: Affine2d::eye(),
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
        self.frame.title(text)
    }

    pub fn xlabel(&mut self, text: &str) -> &mut Text {
        self.frame.xlabel(text)
    }

    pub fn ylabel(&mut self, text: &str) -> &mut Text {
        self.frame.ylabel(text)
    }

    fn default_properties(&mut self) {
        //self.title.font().size(12.);
    }

    ///
    /// Calculate the graph's extents
    /// 
    pub(crate) fn extent(&mut self, canvas: &Canvas) {
        //self.title.update_extent(canvas);
        self.frame.update_extent(canvas);
    }

    ///
    /// Sets the device bounds and propagates to children
    /// 
    pub(crate) fn set_pos(&mut self, pos: &Bounds<Canvas>) {
        /*
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
        */

        self.frame.set_pos(pos);
    }

    pub(crate) fn event(&mut self, renderer: &mut dyn Renderer, event: &CanvasEvent) {
        self.frame.event(renderer, event);
    }

    //
    // TODO: move plots out of graph
    //

    pub(crate) fn draw(&mut self, renderer: &mut dyn Renderer) {
        // self.title.draw(renderer, &self.to_canvas, &self.pos, &self.style);

        self.frame.draw(renderer);
    }

    pub fn add_data_artist(
        &mut self, 
        artist: impl ArtistTrait<Data> + 'static
    ) -> &mut Artist<Data> {
        self.frame.data_mut().artist(artist)
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
