use std::marker::PhantomData;

use essay_plot_base::{Color, Coord, Style};

use crate::{frame::{LayoutArc, FrameId, ArtistId, Data}, artist::Artist};

pub struct PlotOpt {
    layout: LayoutArc,
    frame_id: FrameId,
    artist_id: ArtistId,
}

impl PlotOpt {
    pub(crate) fn new(
        layout: LayoutArc, 
        frame_id: FrameId,
        artist_id: ArtistId,
    ) -> PlotOpt {
        PlotOpt {
            layout,
            frame_id,
            artist_id,
        }
    }

    pub fn color(&mut self, color: impl Into<Color>) -> &mut Self {
        self.layout.borrow_mut()
            .frame_mut(self.frame_id)
            .data_mut()
            .style_mut(self.artist_id)
            .color(color);
        
        self
    }

    pub fn facecolor(&mut self, color: impl Into<Color>) -> &mut Self {
        self.layout.borrow_mut()
            .frame_mut(self.frame_id)
            .data_mut()
            .style_mut(self.artist_id)
            .facecolor(color);
        
        self
    }

    pub fn edgecolor(&mut self, color: impl Into<Color>) -> &mut Self {
        self.layout.borrow_mut()
            .frame_mut(self.frame_id)
            .data_mut()
            .style_mut(self.artist_id)
            .edgecolor(color);
        
        self
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

    pub fn read_style<R>(&self, fun: impl FnOnce(&Style) -> R) -> R {
        fun(self.layout.borrow_mut()
                .frame_mut(self.frame_id)
                .data_mut()
                .style_mut(self.artist_id))
    }

    pub fn write_style<R>(&mut self, fun: impl FnOnce(&mut Style) -> R) -> R {
        fun(self.layout.borrow_mut()
                .frame_mut(self.frame_id)
                .data_mut()
                .style_mut(self.artist_id))
    }
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

