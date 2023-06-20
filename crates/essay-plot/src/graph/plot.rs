use std::marker::PhantomData;

use essay_plot_base::{Color, Coord, JoinStyle, CapStyle, LineStyle};

use crate::{frame::{LayoutArc, FrameId, ArtistId, Data, Frame}, artist::{Artist, PathStyle}};

use super::{Config, ConfigArc};

pub struct PlotOpt {
    //layout: LayoutArc,
    //frame_id: FrameId,
    //artist_id: ArtistId,

    holder: Box<dyn PlotOptTrait>,
}

impl PlotOpt {
    pub(crate) fn new<A: PathStyleArtist + 'static>(
        id: PlotId,
    ) -> PlotOpt {
        PlotOpt {
            holder: Box::new(PlotOptHolder::<A>::new(id))
        }
    }

    pub fn color(&mut self, color: impl Into<Color>) -> &mut Self {
        let color : Color = color.into();

        self.holder.write(&|s| { s.color(color); });
        
        self
    }

    pub fn face_color(&mut self, color: impl Into<Color>) -> &mut Self {
        let color : Color = color.into();

        self.holder.write(&|s| { s.face_color(color); });
        
        self
    }

    pub fn edge_color(&mut self, color: impl Into<Color>) -> &mut Self {
        let color : Color = color.into();

        self.holder.write(&|s| { s.edge_color(color); });
        
        self
    }

    pub fn line_width(&mut self, line_width: f32) -> &mut Self {
        self.holder.write(&|s| { s.line_width(line_width); });

        self
    }

    pub fn line_style(&mut self, line_style: impl Into<LineStyle>) -> &mut Self {
        let line_style : LineStyle = line_style.into();

        self.holder.write(&move |s| { s.line_style(&line_style); });
        
        self
    }

    pub fn join_style(&mut self, join_style: impl Into<JoinStyle>) -> &mut Self {
        let join_style : JoinStyle = join_style.into();

        self.holder.write(&move |s| { s.join_style(join_style); });
        
        self
    }

    pub fn cap_style(&mut self, cap_style: impl Into<CapStyle>) -> &mut Self {
        let cap_style : CapStyle = cap_style.into();

        self.holder.write(&move |s| { s.cap_style(cap_style); });
        
        self
    }

    pub(crate) fn from_id<A>(id: PlotId) -> PlotOpt 
    where
        A: PathStyleArtist
    {
        todo!()
    }
}

struct PlotOptHolder<A: PathStyleArtist> {
    layout: LayoutArc,
    frame_id: FrameId,
    artist_id: ArtistId,
    marker: PhantomData<A>,
}

impl<A: PathStyleArtist> PlotOptHolder<A> {
    fn new(id: PlotId) -> Self {
        Self {
            layout: id.layout,
            frame_id: id.frame_id,
            artist_id: id.artist_id,
            marker: PhantomData
        }
    }
}

impl<A: PathStyleArtist + 'static> PlotOptTrait for PlotOptHolder<A> {
    fn write(&mut self, fun: &dyn Fn(&mut PathStyle)) {
        fun(
            self.layout.borrow_mut()
                .frame_mut(self.frame_id)
                .data_mut()
                .artist_mut::<A>(self.artist_id)
                .style_mut()
        )
    }
}

trait PlotOptTrait {
    fn write(&mut self, fun: &dyn Fn(&mut PathStyle));
}

pub struct PlotId {
    layout: LayoutArc,
    frame_id: FrameId,
    artist_id: ArtistId,
}

impl PlotId {
    pub(crate) fn new(
        layout: LayoutArc, 
        frame_id: FrameId, 
        artist_id: ArtistId
    ) -> Self {
        Self {
            layout,
            frame_id,
            artist_id
        }
    }

    pub(crate) unsafe fn as_ref<M, A>(&self) -> PlotRef<M, A>
    where
        M: Coord,
        A: Artist<M> + 'static
    {
        PlotRef::new(self.layout.clone(), self.frame_id, self.artist_id)
    }
}

pub struct PlotRef<M: Coord, A: Artist<M>> {
    layout: LayoutArc,
    frame_id: FrameId,
    artist_id: ArtistId,
    marker: PhantomData<(M, A)>,
}

impl<M: Coord, A: Artist<M>> PlotRef<M, A> {
    pub(crate) fn new(
        layout: LayoutArc, 
        frame_id: FrameId,
        artist_id: ArtistId,
    ) -> Self {
        Self {
            layout,
            frame_id,
            artist_id,
            marker: PhantomData,
        }
    }
    /*
    pub fn read_style<R>(&self, fun: impl FnOnce(&PathStyle) -> R) -> R {
        fun(self.layout.borrow_mut()
                .frame_mut(self.frame_id)
                .data_mut()
                .style_mut(self.artist_id))
    }

    pub fn write_style<R>(&mut self, fun: impl FnOnce(&mut PathStyle) -> R) -> R {
        fun(self.layout.borrow_mut()
                .frame_mut(self.frame_id)
                .data_mut()
                .style_mut(self.artist_id))
    }
    */
}

impl<A: Artist<Data> + 'static> PlotRef<Data, A> {
    pub fn read_artist<R>(&self, fun: impl FnOnce(&A) -> R) -> R {
        fun(self.layout.borrow_mut()
                .frame_mut(self.frame_id)
                .data_mut()
                .artist(self.artist_id))
    }

    pub fn write_artist<R>(&mut self, fun: impl FnOnce(&mut A) -> R) -> R {
        fun(self.layout.borrow_mut()
                .frame_mut(self.frame_id)
                .data_mut()
                .artist_mut(self.artist_id))
    }
}

pub trait ConfigArtist<M: Coord> : Artist<M> {
    type Opt;
    
    fn config(&mut self, cfg: &ConfigArc, id: PlotId) -> Self::Opt;
}

pub trait PathStyleArtist : Artist<Data> {
    fn style_mut(&mut self) -> &mut PathStyle;
}
