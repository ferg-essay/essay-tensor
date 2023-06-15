use core::fmt;
use std::{rc::Rc, cell::RefCell};

use essay_plot_base::{
    driver::{Renderer}, 
    Bounds, Canvas, CanvasEvent,
};

use crate::{artist::{
    Artist, ArtistStyle,
    Text, ArtAccessor, ArtHolder,
}, frame::{Frame, Data, LayoutArc, FrameId}};

use crate::frame::{Layout};

use super::plot::{PlotOpt, PlotRef};

pub struct Graph {
    id: FrameId,

    layout: LayoutArc,
}

impl Graph {
    pub(crate) fn new(id: FrameId, layout: LayoutArc) -> Self {
        let mut graph = Self {
            id, 
            layout,
        };

        graph.default_properties();

        graph
    }

    #[inline]
    pub fn id(&self) -> FrameId {
        self.id
    }

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

    pub fn add_data_artist(
        &mut self, 
        artist: impl Artist<Data> + 'static
    ) -> PlotOpt {
        let mut layout = self.layout.borrow_mut();
        let frame = layout.frame_mut(self.id);

        let artist_id = frame.data_mut().add_artist(artist);

        PlotOpt::new(self.layout.clone(), frame.id(), artist_id)
    }

    pub fn add_plot<'a, A>(
        &'a mut self, 
        artist: A
    ) -> PlotRef<Data, A>
    where
        A: Artist<Data> + 'static,
    {
        let mut layout = self.layout.borrow_mut();
        let frame = layout.frame_mut(self.id);

        let artist_id = frame.data_mut().add_artist(artist);

        //PlotOpt::new(self.layout.clone(), frame.id(), artist_id)

        PlotRef::new(self.layout.clone(), frame.id(), artist_id)
    }

    pub fn add_artist_holder<'a, A>(
        &'a mut self, 
        artist: A
    ) -> ArtAccessor<'a, Data, A> 
    where
        A: Artist<Data> + 'static
    {
        let mut layout = self.layout.borrow_mut();
        let frame = layout.frame_mut(self.id);

        let rcart = Rc::new(RefCell::new(artist));

        let accessor = ArtAccessor::new(rcart.clone());
        let holder = ArtHolder::new(rcart.clone());

        let _artist_id = frame.data_mut().add_artist(holder);

        //PlotOpt::new(self.layout.clone(), frame.id(), artist_id)
        accessor
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
