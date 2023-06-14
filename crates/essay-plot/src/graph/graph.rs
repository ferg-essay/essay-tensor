use core::fmt;

use essay_plot_base::{
    driver::{Renderer}, 
    Bounds, Canvas, CanvasEvent,
};

use crate::{artist::{
    ArtistTrait, Artist,
    Text,
}, frame::{Frame, Data, LayoutArc, FrameId}};

use crate::frame::{Layout};

use super::plot::PlotOpt;

pub struct Graph {
    id: FrameId,

    // pos: Bounds<Canvas>,

    layout: LayoutArc,

    // frame: Frame,
}

impl Graph {
    pub(crate) fn new(id: FrameId, layout: LayoutArc) -> Self {
        let mut graph = Self {
            id, 
            layout,
            // pos_layout: layout.into(),
            // pos: Bounds::none(),

            //title: Text::new(),
            // frame: Frame::new(),

            //style: Style::default(),

            //to_canvas: Affine2d::eye(),
        };

        graph.default_properties();

        graph
    }

    #[inline]
    pub fn id(&self) -> FrameId {
        self.id
    }

    /*
    pub fn pos(&self) -> &Bounds<Canvas> {
        &self.pos
    }
    */

    pub fn title(&mut self, text: &str) { // -> &mut Text {
        self.layout.borrow_mut().frame_mut(self.id).title(text);
    }

    pub fn xlabel(&mut self, text: &str) { // -> &mut Text {
        self.layout.borrow_mut().frame_mut(self.id).xlabel(text);
    }

    pub fn ylabel(&mut self, text: &str) { // -> &mut Text {
        self.layout.borrow_mut().frame_mut(self.id).ylabel(text);
    }

    fn default_properties(&mut self) {
        //self.title.font().size(12.);
    }
    /*
    #[inline]
    fn frame(&self) -> &Frame {
        self.layout.frame(self.id)
    }

    #[inline]
    fn frame_mut(&mut self) -> &mut Frame {
        self.layout.frame_mut(self.id)
    }
    */

    ///
    /// Calculate the graph's extents
    /// 
    /*
    pub(crate) fn extent(&mut self, canvas: &Canvas) {
        //self.title.update_extent(canvas);
        self.frame.update_extent(canvas);
    }
    */

    ///
    /// Sets the device bounds and propagates to children
    /// 
    /*
    pub(crate) fn set_pos(&mut self, pos: &Bounds<Canvas>) {
        self.frame.set_pos(pos);
    }

    pub(crate) fn event(&mut self, renderer: &mut dyn Renderer, event: &CanvasEvent) {
        self.frame.event(renderer, event);
    }
    */

    //
    // TODO: move plots out of graph
    //

    pub fn add_data_artist(
        &mut self, 
        artist: impl ArtistTrait<Data> + 'static
    ) -> PlotOpt {
        let mut layout = self.layout.borrow_mut();
        let frame = layout.frame_mut(self.id);

        let artist_id = frame.data_mut().artist(artist).id();

        PlotOpt::new(self.layout.clone(), frame.id(), artist_id)
    }
}

impl fmt::Debug for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Graph[{}]({},{}; {}x{})",
            self.id.index(),
            self.layout.borrow().frame(self.id).pos().xmin(),
            self.layout.borrow().frame(self.id).pos().ymin(),
            self.layout.borrow().frame(self.id).pos().width(),
            self.layout.borrow().frame(self.id).pos().height(),
        )
    }
}
