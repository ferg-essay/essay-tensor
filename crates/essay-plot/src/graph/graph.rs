use core::fmt;

use crate::{artist::{
    Artist,
}, frame::{Data, LayoutArc, FrameId, FrameArtist, FrameTextOpt, AxisOpt}};

use super::{plot::{PlotRef, PlotId}, PlotArtist, PlotOpt, style::{PlotStyleArtist, PlotOpt2}};

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

    fn text_opt(&self, artist: FrameArtist) -> FrameTextOpt {
        let layout = self.layout.clone();
        self.layout.borrow().frame(self.id).text_opt(layout, artist)
    }

    pub fn title(&mut self, label: &str) -> FrameTextOpt {
        let mut opt = self.text_opt(FrameArtist::Title);
        opt.label(label);
        opt
    }

    pub fn x(&mut self) -> AxisOpt {
        let layout = self.layout.clone();
        AxisOpt::new(layout, self.id(), FrameArtist::X)
    }

    pub fn y(&mut self) -> AxisOpt {
        let layout = self.layout.clone();
        AxisOpt::new(layout, self.id(), FrameArtist::Y)
    }

    pub fn xlabel(&mut self, label: &str) -> FrameTextOpt {
        let mut opt = self.text_opt(FrameArtist::XLabel);
        opt.label(label);
        opt
    }

    pub fn ylabel(&mut self, label: &str) -> FrameTextOpt {
        let mut opt = self.text_opt(FrameArtist::YLabel);
        opt.label(label);
        opt
    }

    fn default_properties(&mut self) {
        //self.title.font().size(12.);
    }

    // TODO: should there be a plain add_artist that doesn't wrap PlotStyle?

    pub fn add_simple_artist<'a, A>(
        &mut self, 
        artist: A,
    ) -> PlotOpt2
    where
        A: Artist<Data> + 'static
    {
        self.add_plot_artist(PlotStyleArtist::new(artist))
    }

    pub fn add_plot_artist<'a, A>(
        &mut self, 
        artist: A,
    ) -> A::Opt 
    where
        A: PlotArtist<Data> + 'static
    {
        let id = self.layout.borrow_mut()
            .frame_mut(self.id)
            .data_mut()
            .add_artist(artist);

        let plot_id = PlotId::new(
            self.layout.clone(),
            id.frame(),
            id
        );

        self.layout.write(move |layout| {
            let config = layout.config().clone();

            layout
                .frame_mut(id.frame())
                .data_mut()
                .artist_mut::<A>(id)
                .config(&config, plot_id)
        })
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
