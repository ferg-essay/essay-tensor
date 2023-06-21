#[macro_export]
macro_rules! frame_option_struct {
    ($name: ident, $ty: ident, $getter: ident) => {
        pub struct $name {
            layout: $crate::frame::LayoutArc,
            frame_id: $crate::frame::FrameId,
            artist: $crate::frame::FrameArtist,
        }
        
        impl $name {
            pub(crate) fn new(
                layout: $crate::frame::LayoutArc, 
                frame_id: $crate::frame::FrameId,
                artist: $crate::frame::FrameArtist,
            ) -> Self {
                Self {
                    layout,
                    frame_id,
                    artist,
                }
            }
        
            fn write(&mut self, fun: impl FnOnce(&mut $ty)) {
                fun(self.layout.borrow_mut()
                    .frame_mut(self.frame_id)
                    .$getter(self.artist)
                )
            }
        }
    }
}

#[macro_export]
macro_rules! data_artist_option_struct {
    ($name: ident, $ty: ty) => {
        pub struct $name {
            layout: $crate::frame::LayoutArc,
            id: $crate::frame::ArtistId,
        }
        
        impl $name {
            pub(crate) unsafe fn new(
                plot_id: $crate::artist::PlotId,
            ) -> Self {
                Self {
                    layout: plot_id.layout().clone(),
                    id: *plot_id.id(),
                }
            }
        
            fn write(&mut self, fun: impl FnOnce(&mut $ty)) {
                fun(self.layout.borrow_mut()
                    .frame_mut(self.id.frame())
                    .get_data_artist_mut::<$ty>(self.id)
                )
            }
        }
    }
}

#[macro_export]
macro_rules! path_style_options {
    ($field: ident) => {

        pub fn color(&mut self, color: impl Into<essay_plot_base::Color>) -> &mut Self {
            self.write(|ticks| { ticks.$field.color(color); });
            self
        }

        pub fn face_color(&mut self, color: impl Into<essay_plot_base::Color>) -> &mut Self {
            self.write(|ticks| { ticks.$field.face_color(color); });
            self
        }

        pub fn edge_color(&mut self, color: impl Into<essay_plot_base::Color>) -> &mut Self {
            self.write(|ticks| { ticks.$field.edge_color(color); });
            self
        }
    
        pub fn line_width(&mut self, width: f32) -> &mut Self {
            self.write(|ticks| { ticks.$field.line_width(width); });
            self
        }
    
        pub fn line_style(&mut self, style: impl Into<essay_plot_base::LineStyle>) -> &mut Self {
            self.write(|ticks| { ticks.$field.line_style(style); });
            self
        }
    }
}
