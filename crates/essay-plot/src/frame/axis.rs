use essay_plot_base::{TextStyle, Canvas};

use crate::{artist::{PathStyle, Text, Artist}, graph::Config, frame_option_struct, path_style_options};

use super::{
    data_box::DataBox,
    tick_locator::{MaxNLocator, TickLocator}, FrameArtist, tick_formatter::{TickFormatter, Formatter},
};

pub struct Axis {
    show_grid: ShowGrid,

    major: AxisTicks,
    minor: AxisTicks,

    locator: Box<dyn TickLocator>,
    formatter: Box<dyn TickFormatter>,
}

impl Axis {
    pub fn new(cfg: &Config, prefix: &str) -> Self {
        Self {
            //locator: Box::new(LinearLocator::new(None)),
            show_grid: ShowGrid::None,
            major: AxisTicks::new(cfg, &cfg.join(prefix, "major")),
            minor: AxisTicks::new(cfg, &cfg.join(prefix, "minor")),
            locator: Box::new(MaxNLocator::new(None)),
            formatter: Box::new(Formatter::Plain),
        }
    }

    //pub fn ticks(&self, data: &DataBox) -> Tensor<f32> {
    //}

    pub(crate) fn major(&self) -> &AxisTicks {
        &self.major
    }

    pub(crate) fn major_mut(&mut self) -> &mut AxisTicks {
        &mut self.major
    }

    pub(crate) fn minor(&self) -> &AxisTicks {
        &self.minor
    }

    pub(crate) fn minor_mut(&mut self) -> &mut AxisTicks {
        &mut self.minor
    }

    pub fn x_ticks(&self, data: &DataBox) -> Vec<(f32, f32)> {
        let c_width = data.get_pos().width();

        let view = data.get_view_bounds();
        let v_width = view.width();

        if view.is_none() {
            Vec::new()
        } else {
            let (vmin, vmax) = (view.xmin(), view.xmax());
            let (min, max) = self.locator.view_limits(vmin, vmax);

            // self.locator.tick_values(min, max)

            let mut x_vec = Vec::<(f32, f32)>::new();

            for x in self.locator.tick_values(min, max).iter() {
                x_vec.push((*x, ((x - vmin) * c_width / v_width).round()));
            }

            x_vec
        }
    }

    pub fn y_ticks(&self, data: &DataBox) -> Vec<(f32, f32)> {
        let v_height = data.get_view_bounds().height();
        let c_height = data.get_pos().height();

        let view = data.get_view_bounds();

        if view.is_none() {
            Vec::new()
        } else {
            let (vmin, vmax) = (view.ymin(), view.ymax());
            let (min, max) = self.locator.view_limits(vmin, vmax);

            // self.locator.tick_values(min, max)

            let mut y_vec = Vec::<(f32, f32)>::new();

            for y in self.locator.tick_values(min, max).iter() {
                y_vec.push((*y, ((y - vmin) * c_height / v_height).round()));
            }

            y_vec
        }
    }

    pub(crate) fn get_show_grid(&self) -> &ShowGrid {
        &self.show_grid
    }

    fn format(&self, value: f32) -> String {
        self.formatter.format(value)
    }

    pub(crate) fn update(&mut self, canvas: &Canvas) {
        self.major.label_text.update(canvas);
        self.minor.label_text.update(canvas);
    }
}

pub struct AxisTicks {
    grid_style: PathStyle,
    ticks_style: PathStyle,
    label_text: Text,
    size: f32,
    pad: f32,
    locator: Option<Box<dyn TickLocator>>,
    formatter: Option<Box<dyn TickFormatter>>,
}

impl AxisTicks {
    fn new(cfg: &Config, prefix: &str) -> Self {
        let mut ticks = Self {
            grid_style: PathStyle::from_config(cfg, &cfg.join(prefix, "grid")),
            ticks_style: PathStyle::from_config(cfg, &cfg.join(prefix, "ticks")),
            size: match cfg.get_as_type(prefix, "size") {
                Some(size) => size,
                None => 0.,
            },
            pad: match cfg.get_as_type(prefix, "pad") {
                Some(size) => size,
                None => 0.,
            },
            label_text: Text::new(),
            locator: None,
            formatter: None,
        };

        match cfg.get_as_type::<f32>(prefix, "width") {
            Some(width) => { ticks.ticks_style.line_width(width); }
            None => {}
        };

        ticks.label_text.label("0.0");
        
        ticks
    }

    pub(crate) fn grid_style(&self) -> &PathStyle {
        &self.grid_style
    }

    pub(crate) fn tick_style(&self) -> &PathStyle {
        &self.ticks_style
    }

    pub(crate) fn label_style(&self) -> &TextStyle {
        self.label_text.text_style()
    }

    pub(crate) fn label_style_mut(&mut self) -> &mut TextStyle {
        self.label_text.text_style_mut()
    }

    pub(crate) fn grid_style_mut(&mut self) -> &mut PathStyle {
        &mut self.grid_style
    }

    pub(crate) fn format(&self, axis: &Axis, value: f32) -> String {
        match &self.formatter {
            Some(formatter) => {
                formatter.format(value)
            }
            None => { 
                axis.format(value) 
            }
        }
    }

    pub(crate) fn get_size(&self) -> f32 {
        self.size
    }

    pub(crate) fn get_pad(&self) -> f32 {
        self.pad
    }

    pub(crate) fn get_label_height(&self) -> f32 {
        self.label_text.height()
    }

    pub(crate) fn update(&mut self, canvas: &Canvas) {
        self.label_text.update(canvas);
    }

    pub(crate) fn get_text_height_px(&self) -> f32 {
        self.label_text.height()
    }
}

frame_option_struct!(AxisOpt, Axis, get_axis_mut);

impl AxisOpt {
    pub fn show_grid(&mut self, show: impl Into<ShowGrid>) -> &mut Self {
        self.write(|axis| { axis.show_grid = show.into(); });
        self
    }

    pub fn locator(&mut self, locator: impl TickLocator + 'static) -> &mut Self {
        self.write(|axis| { 
            axis.locator = Box::new(locator); 
        });
        self
    }

    pub fn formatter(&mut self, formatter: impl TickFormatter + 'static) -> &mut Self {
        self.write(|axis| { 
            axis.formatter = Box::new(formatter); 
        });
        self
    }

    pub fn major(&self) -> AxisTicksOpt {
        let artist = match self.artist {
            FrameArtist::X => FrameArtist::XMajor,
            FrameArtist::Y => FrameArtist::YMajor,
            _ => panic!("invalid major()")
        };

        AxisTicksOpt::new(self.layout.clone(), self.frame_id, artist)
    }

    pub fn major_grid(&self) -> AxisGridOpt {
        let artist = match self.artist {
            FrameArtist::X => FrameArtist::XMajor,
            FrameArtist::Y => FrameArtist::YMajor,
            _ => panic!("invalid major()")
        };

        AxisGridOpt::new(self.layout.clone(), self.frame_id, artist)
    }
}

frame_option_struct!(AxisGridOpt, AxisTicks, get_ticks_mut);

impl AxisGridOpt {
    path_style_options!(grid_style);
}

frame_option_struct!(AxisTicksOpt, AxisTicks, get_ticks_mut);

impl AxisTicksOpt {
    pub fn locator(&mut self, locator: impl TickLocator + 'static) -> &mut Self {
        self.write(|ticks| { 
            ticks.locator = Some(Box::new(locator)); 
        });
        self
    }

    pub fn formatter(&mut self, formatter: impl TickFormatter + 'static) -> &mut Self {
        self.write(|ticks| { 
            ticks.formatter = Some(Box::new(formatter)); 
        });
        self
    }

    path_style_options!(ticks_style);
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ShowGrid {
    None,
    Major,
    Minor,
    Both,
}

impl ShowGrid {
    pub(crate) fn is_show_major(&self) -> bool {
        match self {
            ShowGrid::None => false,
            ShowGrid::Major => true,
            ShowGrid::Minor => false,
            ShowGrid::Both => true,
        }
    }

    pub(crate) fn is_show_minor(&self) -> bool {
        match self {
            ShowGrid::None => false,
            ShowGrid::Major => false,
            ShowGrid::Minor => true,
            ShowGrid::Both => true,
        }
    }
}

impl From<bool> for ShowGrid {
    fn from(value: bool) -> Self {
        if value {
            Self::Major
        } else {
            Self::None
        }
    }
}
