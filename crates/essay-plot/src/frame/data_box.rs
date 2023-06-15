use core::fmt;

use essay_plot_base::{
    driver::{Renderer}, Style, StyleOpt,
    Bounds, Affine2d, Point, Canvas, Coord, CanvasEvent,
};

use crate::artist::{ArtistStyle, Artist};

use super::plot_container::PlotContainer;

pub struct DataBox {
    pos_canvas: Bounds<Canvas>,

    data_bounds: Bounds<Data>,
    view_bounds: Bounds<Data>,

    artists: PlotContainer<Data>,

    to_canvas: Affine2d,
    style: Style,

    is_modified: bool,
}

impl DataBox {
    pub fn new() -> Self {
        Self {
            pos_canvas: Bounds::none(),

            data_bounds: Bounds::<Data>::unit(),
            view_bounds: Bounds::<Data>::unit(),

            artists: PlotContainer::new(),

            style: Style::default(),

            to_canvas: Affine2d::eye(),
            is_modified: true,
        }
    }

    ///
    /// Sets the canvas bounds
    /// 
    pub(crate) fn set_pos(&mut self, pos: &Bounds<Canvas>) -> &mut Self {
        self.pos_canvas = pos.clone();

        self.to_canvas = self.view_bounds.affine_to(&self.pos_canvas);

        self
    }

    pub fn add_artist(&mut self, artist: impl Artist<Data> + 'static) -> ArtistId {
        let mut artist = artist;

        let bounds = artist.get_extent();

        self.add_data_bounds(&bounds);
        self.add_view_bounds(&bounds);

        let id = self.artists.add_artist(artist);

        id
    }

    fn add_data_bounds(&mut self, bounds: &Bounds<Data>) {
        if self.data_bounds.is_none() {
            self.data_bounds = bounds.clone();
        } else {
            self.data_bounds = self.data_bounds.union(&bounds);
        }
    }

    fn add_view_bounds(&mut self, bounds: &Bounds<Data>) {
        let x_margin = 0.1;
        let y_margin = 0.1;

        let (height, width) = (bounds.height(), bounds.width());

        let (mut xmin, mut xmax) = (bounds.xmin(), bounds.xmax());
        xmin -= x_margin * width;
        xmax += x_margin * width;

        let (mut ymin, mut ymax) = (bounds.ymin(), bounds.ymax());
        ymin -= y_margin * height;
        ymax += y_margin * height;

        let bounds = Bounds::new(Point(xmin, ymin), Point(xmax, ymax));

        if self.view_bounds.is_none() {
            self.view_bounds = bounds;
        } else {
            self.view_bounds = self.view_bounds.union(&bounds);
        }
    }

    fn reset_view(&mut self) {
        let x_margin = 0.1;
        let y_margin = 0.1;

        let data = &self.data_bounds;

        let (height, width) = (data.height(), data.width());

        let (mut xmin, mut xmax) = (data.xmin(), data.xmax());
        xmin -= x_margin * width;
        xmax += x_margin * width;

        let (mut ymin, mut ymax) = (data.ymin(), data.ymax());
        ymin -= y_margin * height;
        ymax += y_margin * height;

        let bounds = Bounds::new(Point(xmin, ymin), Point(xmax, ymax));

        self.view_bounds = bounds;
    }

    pub(crate) fn get_pos(&self) -> &Bounds<Canvas> {
        &self.pos_canvas
    }

    pub(crate) fn get_view_bounds(&self) -> &Bounds<Data> {
        &self.view_bounds
    }

    // true if request redraw
    pub fn event(&mut self, _renderer: &mut dyn Renderer, event: &CanvasEvent) -> bool {
        match event {
            CanvasEvent::ResetView(_) => {
                self.reset_view();
                true
            }
            CanvasEvent::Pan(_p_start, p_last, p_now) => {
                let to_data = self.pos_canvas.affine_to(&self.view_bounds);
                let p0 = to_data.transform_point(*p_last);
                let p1 = to_data.transform_point(*p_now);

                let dx = p0.x() - p1.x();
                let dy = p0.y() - p1.y();

                let view = &self.view_bounds;
                self.view_bounds = Bounds::new(
                    Point(
                        view.x0() + dx,
                        view.y0() + dy,
                    ),
                    Point(
                        view.x1() + dx,
                        view.y1() + dy,
                    )
                );

                true
            },
            CanvasEvent::ZoomBounds(p_start, p_now) => {
                if self.pos_canvas.contains(*p_now) {
                    let to_data = self.pos_canvas.affine_to(&self.view_bounds);
                    let p0 = to_data.transform_point(*p_start);
                    let p1 = to_data.transform_point(*p_now);

                    // let view = &self.view_bounds;
                    // TODO: check min size?
                    self.view_bounds = Bounds::new(p0, p1);
                }

                true
            },
            _ => { false }
        }
    }

    pub(crate) fn style_mut(&mut self, id: ArtistId) -> &mut Style {
        self.artists.style_mut(id)
    }

    pub(crate) fn artist<A>(&self, id: ArtistId) -> &A
    where
        A: Artist<Data> + 'static
    {
        self.artists.artist(id)
    }

    pub(crate) fn artist_mut<A>(&mut self, id: ArtistId) -> &mut A
    where
        A: Artist<Data> + 'static
    {
        self.artists.artist_mut(id)
    }
}

impl Artist<Canvas> for DataBox {
    fn update(&mut self, canvas: &Canvas) {
        self.artists.update(canvas);

        if self.is_modified {
            self.is_modified = false;

            self.data_bounds =self.artists.get_extent();
            /*

            for id in self.artists.iter() {
                let bounds = self.artists.get_extent();

                self.add_data_bounds(&bounds);
                // self.add_view_bounds(&bounds);
            }
            */
    
            self.reset_view();
            println!("Reset: {:?}", self.view_bounds);
        }
    }

    fn get_extent(&mut self) -> Bounds<Canvas> {
        self.pos_canvas.clone()
    }

    fn draw(
        &mut self, 
        renderer: &mut dyn Renderer, 
        _to_canvas: &Affine2d,
        _clip: &Bounds<Canvas>,
        style: &dyn StyleOpt,
    ) {
        //let to_canvas = to_canvas.matmul(&self.to_canvas);
        let to_canvas = &self.to_canvas;
        let style = Style::chain(style, &self.style);

        self.artists.draw(renderer, &to_canvas, &self.pos_canvas, &style);

        // TODO: intersect clip
        //for artist in &mut self.artists {
        //    artist.draw(renderer, &to_canvas, &self.pos_canvas, &style);
        //}
    }
}


impl fmt::Debug for DataBox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DataBox({},{},{}x{})",
            self.view_bounds.xmin(),
            self.view_bounds.ymin(),
            self.view_bounds.width(),
            self.view_bounds.height())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArtistId(pub(crate) usize);

impl ArtistId {
    pub fn index(&self) -> usize {
        self.0
    }

    // TODO: eliminate need for this function
    pub(crate) fn new(index: usize) -> ArtistId {
        ArtistId(index)
    }
}

///
/// Data coordinates
///
#[derive(Clone, Copy, Debug)]
pub struct Data;

impl Coord for Data {
}
