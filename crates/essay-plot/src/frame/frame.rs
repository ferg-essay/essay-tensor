use std::f32::consts::PI;

use essay_plot_base::{
    PathCode, Path, PathOpt,
    driver::{Renderer}, Bounds, Canvas, Affine2d, Point, CanvasEvent, HorizAlign, VertAlign, TextStyle, Color, Clip, 
};
use essay_plot_macro::derive_plot_opt;

use crate::{artist::{patch::{CanvasPatch, Line, PathPatch}, Text, Artist, PathStyle}, graph::Config};

use super::{data_box::DataBox, axis::{Axis, AxisTicks}, tick_formatter::{Formatter, TickFormatter}, layout::FrameId, LayoutArc, Data, legend::Legend};
// self as essay_plot needed for #[derive_plot_opt]
extern crate self as essay_plot;

pub struct Frame {
    id: FrameId,
    
    pos: Bounds<Canvas>,

    to_canvas: Affine2d,

    is_share_x: bool,
    is_share_y: bool,

    path_style: PathStyle,
    // prop_cycle

    data: DataBox,

    title: Text,

    bottom: BottomFrame,
    left: LeftFrame,
    top: TopFrame,
    right: RightFrame,

    is_frame_visible: bool,

    legend: Legend,

    is_stale: bool, 
    aspect_ratio: Option<f32>,
    box_aspect_ratio: Option<f32>,
    // is_visible
    // axis_locator
    // is_axis_below
    // label
    // adjustable (Box vs Data)
    // is_snap
    // transform
    // xbound (min, max)
    // xmargin
    // xscale - linear, log, symlog, logit
    // xticks - sets ticks and labels
    // ybound
    // ylabel
    // ylim
    // ymargin
    // yscale
    // yticks
    // zorder
}

impl Frame {
    pub(crate) fn new(id: FrameId, cfg: &Config) -> Self {
        Self {
            id,

            pos: Bounds::none(),

            data: DataBox::new(id, cfg),

            title: Text::new(),

            bottom: BottomFrame::new(cfg),
            left: LeftFrame::new(cfg),
            top: TopFrame::new(),
            right: RightFrame::new(),

            path_style: PathStyle::default(),

            to_canvas: Affine2d::eye(),

            legend: Legend::new(),

            is_stale: true,
            is_share_x: false,
            is_share_y: false,
            is_frame_visible: true,
            aspect_ratio: None,
            box_aspect_ratio: None,
        }
    }

    #[inline]
    pub fn id(&self) -> FrameId {
        self.id
    }

    pub(crate) fn pos(&self) -> &Bounds<Canvas> {
        &self.pos
    }

    pub(crate) fn update_extent(&mut self, canvas: &Canvas) {
        self.title.update(canvas);

        self.data.update(canvas);

        self.bottom.update_axis(&self.data);
        self.left.update_axis(&self.data);

        self.bottom.update(canvas);
        self.left.update(canvas);
        self.top.update(canvas);
        self.right.update(canvas);

        self.legend.update(canvas);
    }

    ///
    /// Sets the device bounds and propagates to children
    /// 
    /// The position for a frame is the size of the data box. The frame,
    /// axes and titles are relative to the data box.
    /// 
    pub(crate) fn set_pos(&mut self, pos: &Bounds<Canvas>) -> &mut Self {
        self.pos = pos.clone();

        let title = self.title.get_extent();

        // title exists outside the pos bounds
        self.title.set_pos([
            pos.xmin(), pos.ymax(), 
            pos.xmax(), pos.ymax() + title.height()
        ]); 

        let pos_data = Bounds::<Canvas>::new(
            Point(pos.xmin(), pos.ymin()), 
            Point(pos.xmax(), pos.ymax()),
        );

        self.data.set_pos(&pos_data);

        /*
        let pos_bottom = Bounds::<Canvas>::new(
            Point(pos_data.xmin(), pos_data.ymin()),
            Point(pos_data.xmax(), pos_data.ymin()),
        );
        self.bottom.set_pos(pos_bottom);
        */

        let pos_left = Bounds::<Canvas>::new(
            Point(pos_data.xmin(), pos_data.ymin()),
            Point(pos_data.xmin(), pos_data.ymax()),
        );
        self.left.set_pos(pos_left);

        let pos_top = Bounds::<Canvas>::new(
            Point(pos_data.xmin(), pos_data.ymax()),
            Point(pos_data.xmax(), pos_data.ymax()),
        );
        self.top.set_pos(pos_top);

        let pos_right = Bounds::<Canvas>::new(
            Point(pos_data.xmax(), pos_data.ymin()),
            Point(pos_data.xmax(), pos_data.ymax()),
        );
        self.right.set_pos(pos_right);

        let pos_canvas = Bounds::<Canvas>::new(
            Point(pos_data.xmin(), pos_data.ymax()),
            Point(pos_data.xmin(), pos_data.ymax()),
        );
        self.legend.set_pos(pos_canvas);

        self
    }

    pub(crate) fn data(&self) -> &DataBox {
        &self.data
    }

    pub(crate) fn data_mut(&mut self) -> &mut DataBox {
        &mut self.data
    }

    pub(crate) fn text_opt(&self, layout: LayoutArc, artist: FrameArtist) -> FrameTextOpt {
        match artist {
            FrameArtist::Title => FrameTextOpt::new(layout, self.id, artist),
            FrameArtist::XLabel => FrameTextOpt::new(layout, self.id, artist),
            FrameArtist::YLabel => FrameTextOpt::new(layout, self.id, artist),

            _ => panic!("Invalid artist {:?}", artist)
        }
    }

    pub(crate) fn get_text_mut(&mut self, artist: FrameArtist) -> &mut Text {
        match artist {
            FrameArtist::Title => &mut self.title,
            FrameArtist::XLabel => &mut self.bottom.title,
            FrameArtist::YLabel => &mut self.left.title,

            _ => panic!("Invalid text {:?}", artist)
        }
    }

    pub(crate) fn get_axis_mut(&mut self, artist: FrameArtist) -> &mut Axis {
        match artist {
            FrameArtist::X => &mut self.bottom.axis,
            FrameArtist::Y => &mut self.left.axis,

            _ => panic!("Invalid axis {:?}", artist)
        }
    }

    pub(crate) fn get_ticks_mut(&mut self, artist: FrameArtist) -> &mut AxisTicks {
        match artist {
            FrameArtist::XMajor => self.bottom.axis.major_mut(),
            FrameArtist::XMinor => self.bottom.axis.minor_mut(),
            FrameArtist::YMajor => self.left.axis.major_mut(),
            FrameArtist::YMinor => self.left.axis.minor_mut(),

            _ => panic!("Invalid axis-texts {:?}", artist)
        }
    }

    pub(crate) fn event(&mut self, renderer: &mut dyn Renderer, event: &CanvasEvent) {
        if self.data.get_pos().contains(event.point()) {
            if self.data.event(renderer, event) {
                self.left.update_axis(&self.data);
                self.bottom.update_axis(&self.data);

                renderer.request_redraw(&self.pos);
            };
        }
    }

    pub(crate) fn draw(&mut self, renderer: &mut dyn Renderer) {
        let clip = Clip::from(&self.pos);

        self.title.draw(renderer, &self.to_canvas, &clip, &self.path_style);

        self.bottom.draw(renderer, &self.data, &self.to_canvas, &clip, &self.path_style);
        self.left.draw(renderer, &self.data, &self.to_canvas, &clip, &self.path_style);

        self.top.draw(renderer, &self.to_canvas, &clip, &self.path_style);
        self.right.draw(renderer,  &self.to_canvas, &clip, &self.path_style);

        // TODO: grid order
        self.data.draw(renderer, &self.to_canvas, &clip, &self.path_style);

        self.legend.draw(renderer, &self.to_canvas, &clip, &self.path_style);
    }

    pub fn title(&mut self, text: &str) -> &mut Text {
        self.title.label(text);

        &mut self.title
    }

    pub fn xlabel(&mut self, text: &str) -> &mut Text {
        self.bottom.title(text)
    }

    pub fn ylabel(&mut self, text: &str) -> &mut Text {
        self.left.label(text)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FrameArtist {
    Title,
    X,
    Y,
    XMajor,
    XMinor,
    YMajor,
    YMinor,
    XLabel,
    YLabel,
}

pub struct FrameSizes {
    line_width: f32,

    title_pad: f32,
    label_pad: f32,
}

impl FrameSizes {
    fn new(cfg: &Config) -> Self {

        // frame.title_pad: 6.0
        // frame.label_size: medium
        // frame.label_pad: 4.0
        // frame.title_size: large
        // xaxis.major.size: 3.5
        // xaxis.major.pad: 3.5

        Self {
            line_width: 1.,

            title_pad: match cfg.get_as_type("frame", "title_pad") {
                Some(pad) => pad,
                None => 0.,
            },

            label_pad: match cfg.get_as_type("frame", "label_pad") {
                Some(pad) => pad,
                None => 0.,
            },
        }
    }
}

//
// Top Frame
//

pub struct TopFrame {
    bounds: Bounds<Canvas>,
    pos: Bounds<Canvas>,
    spine: Option<CanvasPatch>,
}

impl TopFrame {
    pub fn new() -> Self {
        Self {
            bounds: Bounds::new(Point(0., 0.), Point(0., 20.)),
            pos: Bounds::none(),
            spine: Some(CanvasPatch::new(Line::new(Point(0., 0.), Point(1., 0.)))),
        }
    }

    pub fn set_pos(&mut self, pos: Bounds<Canvas>) {
        self.pos = pos.clone();

        if let Some(spine) = &mut self.spine {
            spine.set_pos(Bounds::new(
                Point(pos.xmin(), pos.ymin()),
                Point(pos.xmax(), pos.ymin() + 1.),
            ))
        }
    }
}

impl Artist<Canvas> for TopFrame {
    fn update(&mut self, _canvas: &Canvas) {
    }
    
    fn get_extent(&mut self) -> Bounds<Canvas> {
        self.bounds.clone()
    }

    fn draw(
        &mut self, 
        renderer: &mut dyn Renderer,
        to_canvas: &Affine2d,
        clip: &Clip,
        style: &dyn PathOpt,
    ) {
        if let Some(patch) = &mut self.spine {
            patch.draw(renderer, to_canvas, clip, style);
        }
        
    }
}

//
// Bottom frame
//

pub struct BottomFrame {
    sizes: FrameSizes,

    spine: Option<CanvasPatch>,

    axis: Axis,
    major_ticks: Vec<f32>,
    major_labels: Vec<String>,

    title: Text,
}

impl BottomFrame {
    pub fn new(cfg: &Config) -> Self {
        let mut frame = Self {
            sizes: FrameSizes::new(cfg),
            spine: Some(CanvasPatch::new(Line::new(Point(0., 0.), Point(1., 0.)))),
            axis: Axis::new(cfg, "x_axis"),

            major_ticks: Vec::new(),
            major_labels: Vec::new(),

            title: Text::new(),
        };

        frame.axis.major_mut().label_style_mut().valign(VertAlign::Top);
        frame.axis.minor_mut().label_style_mut().valign(VertAlign::Top);
        frame.title.text_style_mut().valign(VertAlign::Top);

        frame
    }

    pub fn update_axis(&mut self, data: &DataBox) {
        self.major_ticks = Vec::new();
        self.major_labels = Vec::new();

        let xmin = data.get_view_bounds().xmin();
        let xmax = data.get_view_bounds().xmax();

        for (xv, _) in self.axis.x_ticks(data) {
            if xmin <= xv && xv <= xmax {
                self.major_ticks.push(xv);
                self.major_labels.push(self.axis.major().format(&self.axis, xv));
            };
        }
    }

    fn draw(
        &mut self, 
        renderer: &mut dyn Renderer,
        data: &DataBox,
        to_canvas: &Affine2d,
        clip: &Clip,
        style: &dyn PathOpt,
    ) {
        let pos = data.get_pos();

        if let Some(patch) = &mut self.spine {
            patch.set_pos(Bounds::new(
                Point(pos.xmin(), pos.ymin() - 1.),
                Point(pos.xmax(), pos.ymin()),
            ));

            patch.draw(renderer, to_canvas, clip, style);
        }

        self.draw_ticks(renderer, &data, clip, style);

        let mut y = data.get_pos().ymin();
        y -= renderer.to_px(self.axis.major().get_size());
        y -= renderer.to_px(self.axis.major().get_pad());
        y -= self.axis.major().get_label_height();
        y -= renderer.to_px(self.sizes.label_pad);

        self.title.set_pos(Bounds::new(
            Point(data.get_pos().xmin(), y),
            Point(data.get_pos().xmax(), y),
        ));

        self.title.draw(renderer, to_canvas, clip, style);
    }

    fn draw_ticks(
        &mut self, 
        renderer: &mut dyn Renderer, 
        data: &DataBox,
        clip: &Clip,
        style: &dyn PathOpt,
    ) {
        let pos = &data.get_pos();

        let yv = data.get_view_bounds().ymin();
        let to_canvas = data.get_canvas_transform();

        for (xv, label) in self.major_ticks.iter().zip(self.major_labels.iter()) {
            let point = to_canvas.transform_point(Point(*xv, yv));

            let x = point.x();
            let mut y = pos.ymin();
            let major = self.axis.major();

            // Grid
            if self.axis.get_show_grid().is_show_major() {
                let style = major.grid_style().push(style);
                // grid
                let grid = Path::<Canvas>::new(vec![
                    PathCode::MoveTo(Point(x, pos.ymin())),
                    PathCode::LineTo(Point(x, pos.ymax())),
                ]);

                renderer.draw_path(&style, &grid, to_canvas, clip).unwrap();
            }

            // Tick
            {
                let style = major.tick_style().push(style);
                let tick_length = renderer.to_px(major.get_size());
                
                let tick = Path::<Canvas>::new(vec![
                    PathCode::MoveTo(Point(x, y)),
                    PathCode::LineTo(Point(x, y - tick_length)),
                ]);

                renderer.draw_path(&style, &tick, to_canvas, clip).unwrap();

                y -= tick_length;
                y -= renderer.to_px(major.get_pad());
            }

            // Label
            renderer.draw_text(Point(x, y), label, 0., style, major.label_style(), clip).unwrap();
        }
    }

    fn title(&mut self, text: &str) -> &mut Text {
        self.title.label(text)
    }

    fn update(&mut self, canvas: &Canvas) {
        self.title.update(canvas);
        self.axis.update(canvas);
    }
}

//
// Left frame
//

pub struct LeftFrame {
    extent: Bounds<Canvas>,
    pos: Bounds<Canvas>,

    sizes: FrameSizes,

    spine: Option<CanvasPatch>,

    axis: Axis,
    ticks: Vec<Box<dyn Artist<Canvas>>>,

    is_grid_major: bool,
    style_major: PathStyle,
    grid_major: Vec<Box<dyn Artist<Canvas>>>,
    style_minor: PathStyle,
    grid_minor: Vec<Box<dyn Artist<Canvas>>>,

    major_ticks: Vec<f32>,
    major_labels: Vec<String>,

    title: Text,
}

impl LeftFrame {
    pub fn new(cfg: &Config) -> Self {
        let mut style_major = PathStyle::new();
        style_major.line_width(1.0);
        style_major.color(0xbfbfbf);
        let mut style_minor = PathStyle::new();
        style_minor.line_width(1.);
        style_minor.color(0x404040);

        let mut label = Text::new();
        label.angle(PI / 2.);

        let mut frame = Self {
            extent: Bounds::new(Point(0., 0.), Point(20., 0.)),
            pos: Bounds::none(),

            sizes: FrameSizes::new(cfg),

            spine: Some(CanvasPatch::new(Line::new(Point(0., 0.), Point(0., 1.)))),
            axis: Axis::new(cfg, "y_axis"),
            ticks: Vec::new(),

            is_grid_major: false,
            grid_major: Vec::new(),
            style_major,
            grid_minor: Vec::new(),
            style_minor,

            major_ticks: Vec::new(),
            major_labels: Vec::new(),

            title: label,
        };

        frame.axis.major_mut().label_style_mut().valign(VertAlign::Center);
        frame.axis.major_mut().label_style_mut().halign(HorizAlign::Right);
        frame.title.text_style_mut().valign(VertAlign::BaselineBottom);

        frame
    }

    pub fn set_pos(&mut self, pos: Bounds<Canvas>) {
        self.pos = pos.clone();

        if let Some(spine) = &mut self.spine {
            spine.set_pos(Bounds::new(
                Point(pos.xmax() - 1., pos.ymin()),
                Point(pos.xmax(), pos.ymax()),
            ))
        }

        let x0 = pos.xmax();
        self.title.set_pos(Bounds::new(
            Point(x0 - self.title.height(), pos.ymid()),
            Point(x0, pos.ymid())
        ));
    }

    pub fn update_axis(&mut self, data: &DataBox) {
        self.major_ticks = Vec::new();
        self.major_labels = Vec::new();

        let ymin = data.get_view_bounds().ymin();
        let ymax = data.get_view_bounds().ymax();

        for (yv, _) in self.axis.y_ticks(data) {
            if ymin <= yv && yv <= ymax {
                self.major_ticks.push(yv);
                self.major_labels.push(self.axis.major().format(&self.axis, yv));
            };
        }
    }

    fn draw(
        &mut self, 
        renderer: &mut dyn Renderer,
        data: &DataBox,
        to_canvas: &Affine2d,
        clip: &Clip,
        style: &dyn PathOpt,
    ) {
        let pos = data.get_pos();

        if let Some(patch) = &mut self.spine {
            patch.set_pos(Bounds::new(
                Point(pos.xmin() - 1., pos.ymin()),
                Point(pos.xmin(), pos.ymax()),
            ));

            patch.draw(renderer, to_canvas, clip, style);
        }

        self.draw_ticks(renderer, &data, clip, style);

        let width = self.major_labels.iter().map(|s| s.len()).max().unwrap();
        
        let mut x = data.get_pos().xmin();
        x -= renderer.to_px(self.axis.major().get_size());
        x -= renderer.to_px(self.axis.major().get_pad());
        x -= 0.5 * width as f32 * self.axis.major().get_label_height();
        x -= renderer.to_px(self.sizes.label_pad);

        self.title.set_pos(Bounds::new(
            Point(x, data.get_pos().ymid()),
            Point(x, data.get_pos().ymid()),
        ));

        self.title.draw(renderer, to_canvas, clip, style);
    }

    fn draw_ticks(
        &mut self, 
        renderer: &mut dyn Renderer, 
        data: &DataBox,
        clip: &Clip,
        style: &dyn PathOpt,
    ) {
        let pos = &data.get_pos();

        let xv = data.get_view_bounds().xmin();
        let to_canvas = data.get_canvas_transform();

        for (yv, label) in self.major_ticks.iter().zip(self.major_labels.iter()) {
            let point = to_canvas.transform_point(Point(xv, *yv));

            let y = point.y();
            let mut x = pos.xmin();
            let major = self.axis.major();

            // Grid
            if self.axis.get_show_grid().is_show_major() {
                let style = major.grid_style().push(style);
                // grid
                let grid = Path::<Canvas>::new(vec![
                    PathCode::MoveTo(Point(pos.xmin(), y)),
                    PathCode::LineTo(Point(pos.xmax(), y)),
                ]);

                renderer.draw_path(&style, &grid, to_canvas, clip).unwrap();
            }

            // Tick
            {
                let style = major.tick_style().push(style);
                let tick_length = renderer.to_px(major.get_size());
                
                let tick = Path::<Canvas>::new(vec![
                    PathCode::MoveTo(Point(x - tick_length, y)),
                    PathCode::LineTo(Point(x, y)),
                ]);

                renderer.draw_path(&style, &tick, to_canvas, clip).unwrap();

                x -= tick_length;
                x -= renderer.to_px(major.get_pad());
            }

            // Label
            renderer.draw_text(Point(x, y), label, 0., style, major.label_style(), clip).unwrap();
        }
    }

    fn label(&mut self, text: &str) -> &mut Text {
        self.title.label(text)
    }
}

impl Artist<Canvas> for LeftFrame {
    fn update(&mut self, canvas: &Canvas) {
        self.title.update(canvas);
        self.axis.update(canvas);

        let mut width = self.sizes.line_width;
        //width += self.sizes.major_size;
        //width += self.sizes.major_gap;

        //self.sizes.tick_text_height_px = 5. * canvas.scale_factor() * 14.;
        //width += self.sizes.tick_text_height_px;
        //width += self.sizes.margin;
        
        self.extent = Bounds::new(Point(0., 0.), Point(width, 0.))
    }

    fn get_extent(&mut self) -> Bounds<Canvas> {
        self.extent.clone()
    }

    fn draw(
        &mut self, 
        renderer: &mut dyn Renderer,
        to_canvas: &Affine2d,
        clip: &Clip,
        style: &dyn PathOpt,
    ) {
        self.title.draw(renderer, to_canvas, clip, style);
        
        if self.axis.get_show_grid().is_show_major() {
            let style = self.axis.major().grid_style().push(&self.style_major);

            for grid in &mut self.grid_major {
                grid.draw(renderer, to_canvas, clip, &style);
            }
        }

        if self.axis.get_show_grid().is_show_minor() {
            for grid in &mut self.grid_minor {
                grid.draw(renderer, to_canvas, clip, &self.style_minor);
            }
        }

        let tick_style = self.axis.major().tick_style().push(style);
        
        for tick in &mut self.ticks {
            tick.draw(renderer, to_canvas, clip, &tick_style);
        }

        if let Some(patch) = &mut self.spine {
            patch.draw(renderer, to_canvas, clip, style);
        }
    }
}

//
// Right frame
//

pub struct RightFrame {
    bounds: Bounds<Canvas>,
    pos: Bounds<Canvas>,
    spine: Option<CanvasPatch>,
}

impl RightFrame {
    pub fn new() -> Self {
        Self {
            bounds: Bounds::new(Point(0., 0.), Point(20., 0.)),
            pos: Bounds::none(),
            spine: Some(CanvasPatch::new(Line::new(Point(0., 0.), Point(0., 1.)))),
        }
    }

    pub fn set_pos(&mut self, pos: Bounds<Canvas>) {
        self.pos = pos.clone();

        if let Some(spine) = &mut self.spine {
            spine.set_pos(Bounds::new(
                Point(pos.xmin(), pos.ymin()),
                Point(pos.xmin() + 1., pos.ymax()),
            ))
        }
    }
}

impl Artist<Canvas> for RightFrame {
    fn update(&mut self, _canvas: &Canvas) {
    }
    
    fn get_extent(&mut self) -> Bounds<Canvas> {
        self.bounds.clone()
    }

    fn draw(
        &mut self, 
        renderer: &mut dyn Renderer,
        to_canvas: &Affine2d,
        clip: &Clip,
        style: &dyn PathOpt,
    ) {
        if let Some(patch) = &mut self.spine {
            patch.draw(renderer, to_canvas, clip, style);
        }
    }
}

pub struct FrameTextOpt {
    layout: LayoutArc,
    id: FrameId,
    artist: FrameArtist,
}

impl FrameTextOpt {
    fn new(layout: LayoutArc, id: FrameId, artist: FrameArtist) -> Self {
        Self {
            layout,
            id,
            artist,
        }
    }

    fn write(&mut self, fun: impl FnOnce(&mut Text)) {
        fun(self.layout.borrow_mut().frame_mut(self.id).get_text_mut(self.artist))
    }

    pub fn label(&mut self, label: &str) -> &mut Self {
        self.write(|text| { text.label(label); });
        self
    }

    pub fn color(&mut self, color: impl Into<Color>) -> &mut Self {
        self.write(|text| { text.path_style_mut().color(color); });
        self
    }

    pub fn size(&mut self, size: f32) -> &mut Self {
        self.write(|text| { text.text_style_mut().size(size); });
        self
    }
}
