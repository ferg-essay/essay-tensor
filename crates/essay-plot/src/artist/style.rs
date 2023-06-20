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
