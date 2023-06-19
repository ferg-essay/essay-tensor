use core::fmt;
use std::str::FromStr;

use essay_plot_base::{Color, JoinStyle, CapStyle, PathOpt, LineStyle, TextureId};

use crate::graph::Config;


#[derive(Clone)]
pub struct PathStyle {
    color: Option<Color>,
    face_color: Option<Color>,
    edge_color: Option<Color>,

    line_width: Option<f32>,
    join_style: Option<JoinStyle>,
    cap_style: Option<CapStyle>,

    line_style: Option<LineStyle>,
    gap_color: Option<Color>,
}

impl PathStyle {
    pub fn new() -> PathStyle {
        PathStyle::default()
    }

    pub(crate) fn from_config(cfg: &Config, prefix: &str) -> PathStyle {
        let mut style = PathStyle::default();

        style.color = cfg.get_as_type(prefix, "color");
        style.face_color = cfg.get_as_type(prefix, "face_color");
        style.edge_color = cfg.get_as_type(prefix, "edge_color");
        style.gap_color = cfg.get_as_type(prefix, "gap_color");
        style.line_width = cfg.get_as_type(prefix, "line_width");
        style.line_style = cfg.get_as_type(prefix, "line_style");
        style.join_style = cfg.get_as_type(prefix, "join_style");
        style.cap_style = cfg.get_as_type(prefix, "cap_style");
        style
    }

    pub fn color(&mut self, color: impl Into<Color>) -> &mut Self {
        // TODO: is color a default or an assignment?
        self.color = Some(color.into());

        self
    }

    pub fn face_color(&mut self, color: impl Into<Color>) -> &mut Self {
        self.face_color = Some(color.into());

        self
    }

    pub fn edge_color(&mut self, color: impl Into<Color>) -> &mut Self {
        self.edge_color = Some(color.into());

        self
    }

    pub fn line_style(&mut self, line_style: impl Into<LineStyle>) -> &mut Self {
        self.line_style = Some(line_style.into());

        self
    }

    pub fn line_width(&mut self, linewidth: f32) -> &mut Self {
        assert!(linewidth > 0.);

        self.line_width = Some(linewidth);

        self
    }

    pub fn join_style(&mut self, joinstyle: impl Into<JoinStyle>) -> &mut Self {
        self.join_style = Some(joinstyle.into());

        self
    }

    pub fn cap_style(&mut self, capstyle: impl Into<CapStyle>) -> &mut Self {
        self.cap_style = Some(capstyle.into());

        self
    }
}

fn color_from_config(cfg: &Config, prefix: &str, name: &str) -> Option<Color> {
    match cfg.get_with_prefix(prefix, name) {
        Some(value) => Some(Color::from(value.as_str())),
        None => None,
    }
}

fn from_config<T: FromStr>(cfg: &Config, prefix: &str, name: &str) -> Option<T>
where <T as FromStr>::Err : fmt::Debug
{
    match cfg.get_with_prefix(prefix, name) {
        Some(value) => Some(value.parse::<T>().unwrap()),
        None => None,
    }
}

impl PathOpt for PathStyle {
    fn get_face_color(&self) -> &Option<Color> {
        match &self.face_color {
            Some(_color) => &self.face_color,
            None => &self.color,
        }
    }

    fn get_edge_color(&self) -> &Option<Color> {
        match &self.edge_color {
            Some(_color) => &self.edge_color,
            None => &self.color,
        }
    }

    fn get_line_width(&self) -> &Option<f32> {
        &self.line_width
    }

    fn get_join_style(&self) -> &Option<JoinStyle> {
        &self.join_style
    }

    fn get_cap_style(&self) -> &Option<CapStyle> {
        &self.cap_style
    }

    fn get_line_style(&self) -> &Option<LineStyle> {
        &self.line_style
    }

    fn get_alpha(&self) -> &Option<f32> {
        todo!()
    }

    fn get_texture(&self) -> &Option<essay_plot_base::TextureId> {
        todo!()
    }
}

impl Default for PathStyle {
    fn default() -> Self {
        Self { 
            color: Default::default(), 
            face_color: Default::default(), 
            edge_color: Default::default(), 
            line_width: Default::default(), 
            join_style: Default::default(),
            cap_style: Default::default(),
            line_style: Default::default(),
            gap_color: Default::default(),
        }
    }
}

pub struct StyleCycle {
    fill_colors: Vec<Option<Color>>,
    edge_colors: Vec<Option<Color>>,
    line_widths: Vec<Option<f32>>,
    line_styles: Vec<Option<LineStyle>>,
}

impl StyleCycle {
    pub fn new() -> Self {
        Self {
            fill_colors: Vec::new(),
            edge_colors: Vec::new(),
            line_widths: Vec::new(),
            line_styles: Vec::new(),
        }
    }
    pub fn push<'a>(
        &'a self, 
        prev: &'a dyn PathOpt, 
        index: usize
    ) -> PropCycleChain<'a> {
        PropCycleChain::new(self, prev, index)
    }

    pub fn fill_colors(&mut self, colors: impl Into<Colors>) -> &mut Self {
        let colors: Colors = colors.into();

        self.fill_colors = colors.color_cycle;

        self
    }

    pub fn edge_colors(&mut self, colors: impl Into<Colors>) -> &mut Self {
        let colors: Colors = colors.into();

        self.edge_colors = colors.color_cycle;

        self
    }

    pub fn colors(&mut self, colors: impl Into<Colors>) -> &mut Self {
        let colors: Colors = colors.into();

        self.fill_colors = colors.color_cycle.clone();
        self.edge_colors = colors.color_cycle;

        self
    }

    pub fn line_styles(&mut self, line_styles: impl Into<LineStyles>) -> &mut Self {
        let cycle: LineStyles = line_styles.into();

        self.line_styles = cycle.cycle;

        self
    }

    pub fn line_widths(&mut self, widths: &[f32]) -> &mut Self {
        let mut vec: Vec<Option<f32>> = Vec::new();

        for width in widths {
            vec.push(Some(*width));
        }

        self.line_widths = vec;

        self
    }

    fn is_fill_color_set(&self) -> bool {
        self.fill_colors.len() > 0
    }

    fn get_fill_color(&self, index: usize) -> &Option<Color> {
        let len = self.fill_colors.len();
        assert!(len > 0);

        &self.fill_colors[index % self.fill_colors.len()]
    }

    fn is_edge_color_set(&self) -> bool {
        self.edge_colors.len() > 0
    }

    fn get_edge_color(&self, index: usize) -> &Option<Color> {
        let len = self.edge_colors.len();
        assert!(len > 0);
        
        &self.edge_colors[index % self.edge_colors.len()]
    }

    fn is_line_width_set(&self) -> bool {
        self.line_widths.len() > 0
    }

    fn get_line_width(&self, index: usize) -> &Option<f32> {
        let len = self.line_widths.len();
        assert!(len > 0);
        
        &self.line_widths[index % self.line_widths.len()]
    }

    fn is_line_style_set(&self) -> bool {
        self.line_styles.len() > 0
    }

    fn get_line_style(&self, index: usize) -> &Option<LineStyle> {
        let len = self.line_styles.len();
        assert!(len > 0);
        
        &self.line_styles[index % self.line_styles.len()]
    }
}

pub struct PropCycleChain<'a> {
    cycle: &'a StyleCycle,
    prev: &'a dyn PathOpt,
    index: usize,
}

impl<'a> PropCycleChain<'a> {
    fn new(
        cycle: &'a StyleCycle,
        prev: &'a dyn PathOpt,
        index: usize,
    ) -> Self {
        Self {
            prev,
            cycle,
            index,
        }
    }
}

impl PathOpt for PropCycleChain<'_> {
    fn get_face_color(&self) -> &Option<Color> {
        if self.cycle.is_fill_color_set() {
            self.cycle.get_fill_color(self.index)
        } else {
            self.prev.get_face_color()
        }
    }

    fn get_edge_color(&self) -> &Option<Color> {
        if self.cycle.is_edge_color_set() {
            self.cycle.get_edge_color(self.index)
        } else {
            self.prev.get_edge_color()
        }
    }

    fn get_line_style(&self) -> &Option<LineStyle> {
        if self.cycle.is_line_style_set() {
            self.cycle.get_line_style(self.index)
        } else {
            self.prev.get_line_style()
        }
    }

    fn get_line_width(&self) -> &Option<f32> {
        if self.cycle.is_line_width_set() {
            self.cycle.get_line_width(self.index)
        } else {
            self.prev.get_line_width()
        }
    }

    fn get_join_style(&self) -> &Option<JoinStyle> {
        self.prev.get_join_style()
    }

    fn get_cap_style(&self) -> &Option<CapStyle> {
        self.prev.get_cap_style()
    }

    fn get_alpha(&self) -> &Option<f32> {
        self.prev.get_alpha()
    }

    fn get_texture(&self) -> &Option<TextureId> {
        self.prev.get_texture()
    }
}

pub struct Colors {
    color_cycle: Vec<Option<Color>>,
}

impl<const N: usize> From<[Color; N]> for Colors {
    fn from(value: [Color; N]) -> Self {
        let mut vec = Vec::new();

        for color in value {
            vec.push(Some(color));
        }

        Self { color_cycle: vec }
    }
}

impl From<&[Color]> for Colors {
    fn from(value: &[Color]) -> Self {
        let mut vec = Vec::new();

        for color in value {
            vec.push(Some(color.clone()));
        }

        Self { color_cycle: vec }
    }
}

impl<const N: usize> From<[&str; N]> for Colors {
    fn from(value: [&str; N]) -> Self {
        let mut vec = Vec::new();

        for name in value {
            vec.push(Some(Color::from(name)));
        }

        Self { color_cycle: vec }
    }
}

impl From<&[&str]> for Colors {
    fn from(value: &[&str]) -> Self {
        let mut vec = Vec::new();

        for name in value {
            vec.push(Some(Color::from(*name)));
        }

        Self { color_cycle: vec }
    }
}

impl<const N: usize> From<[u32; N]> for Colors {
    fn from(value: [u32; N]) -> Self {
        let mut vec = Vec::new();

        for name in value {
            vec.push(Some(Color::from(name)));
        }

        Self { color_cycle: vec }
    }
}

impl From<&[u32]> for Colors {
    fn from(value: &[u32]) -> Self {
        let mut vec = Vec::new();

        for rgb in value {
            vec.push(Some(Color::from(*rgb)));
        }

        Self { color_cycle: vec }
    }
}


pub struct LineStyles {
   cycle: Vec<Option<LineStyle>>,
}

impl<const N: usize> From<[LineStyle; N]> for LineStyles {
    fn from(value: [LineStyle; N]) -> Self {
        let mut vec = Vec::new();

        for style in value {
            vec.push(Some(style));
        }

        Self { cycle: vec }
    }
}

impl From<&[LineStyle]> for LineStyles {
    fn from(value: &[LineStyle]) -> Self {
        let mut vec = Vec::new();

        for style in value {
            vec.push(Some(style.clone()));
        }

        Self { cycle: vec }
    }
}

impl<const N: usize> From<[&str; N]> for LineStyles {
    fn from(value: [&str; N]) -> Self {
        let mut vec = Vec::new();

        for name in value {
            vec.push(Some(LineStyle::from(name)));
        }

        Self { cycle: vec }
    }
}

impl From<&[&str]> for LineStyles {
    fn from(value: &[&str]) -> Self {
        let mut vec = Vec::new();

        for name in value {
            vec.push(Some(LineStyle::from(*name)));
        }

        Self { cycle: vec }
    }
}

