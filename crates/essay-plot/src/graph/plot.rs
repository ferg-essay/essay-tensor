use essay_plot_base::Color;

use crate::frame::{LayoutArc, FrameId, ArtistId};

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
            .artist_mut(self.artist_id)
            .color(color);
        
        self
    }

    pub fn facecolor(&mut self, color: impl Into<Color>) -> &mut Self {
        self.layout.borrow_mut()
            .frame_mut(self.frame_id)
            .data_mut()
            .artist_mut(self.artist_id)
            .facecolor(color);
        
        self
    }

    pub fn edgecolor(&mut self, color: impl Into<Color>) -> &mut Self {
        self.layout.borrow_mut()
            .frame_mut(self.frame_id)
            .data_mut()
            .artist_mut(self.artist_id)
            .edgecolor(color);
        
        self
    }
}
