use core::fmt;

use essay_plot_base::{Color, JoinStyle, CapStyle, PathOpt, LineStyle};

use crate::graph::Config;

use super::{Markers};

pub trait PathStyleOpt : PathOpt {
    fn get_marker(&self) -> &Option<Markers>;
}

#[derive(Clone)]
pub struct PathStyle {
    color: Option<Color>,
    face_color: Option<Color>,
    edge_color: Option<Color>,

    line_width: Option<f32>,
    join_style: Option<JoinStyle>,
    cap_style: Option<CapStyle>,

    line_style: Option<LineStyle>,
    alpha: Option<f32>,


    gap_color: Option<Color>,

    marker: Option<Markers>,
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
        style.alpha = cfg.get_as_type(prefix, "alpha");
        style.marker = cfg.get_as_type(prefix, "marker");
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

    pub fn alpha(&mut self, alpha: f32) -> &mut Self {
        self.alpha = Some(alpha);

        self
    }

    pub fn marker(&mut self, marker: impl Into<Markers>) -> &mut Self {
        self.marker = Some(marker.into());

        self
    }
}

impl fmt::Debug for PathStyle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut fmt = f.debug_struct("PathStyle");

        if let Some(color) = &self.color {
            fmt.field("color", color);
        }

        if let Some(face_color) = &self.face_color {
            fmt.field("face_color", face_color);
        }

        if let Some(edge_color) = &self.edge_color {
            fmt.field("edge_color", edge_color);
        }
        
        if let Some(line_width) = &self.line_width {
            fmt.field("line_width", line_width);
        }
        
        if let Some(join_style) = &self.join_style {
            fmt.field("join_style", join_style);
        }

        if let Some(cap_style) = &self.cap_style {
            fmt.field("cap_style", cap_style);
        }
        
        if let Some(line_style) = &self.line_style {
            fmt.field("line_style", line_style);
        }
        
        if let Some(alpha) = &self.alpha {
            fmt.field("alpha", alpha);
        }
        
        if let Some(gap_color) = &self.gap_color {
            fmt.field("gap_color", gap_color);
        }
        
        if let Some(marker) = &self.marker {
            fmt.field("marker", marker);
        }
        
        fmt.finish()
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
        &self.alpha
    }

    fn get_texture(&self) -> &Option<essay_plot_base::TextureId> {
        todo!()
    }
}

impl PathStyleOpt for PathStyle {
    fn get_marker(&self) -> &Option<Markers> {
        &self.marker
    }
}

impl Default for PathStyle {
    fn default() -> Self {
        Self { 
            color: None,
            face_color: None,
            edge_color: None,
            line_width: None,
            join_style: None,
            cap_style: None,
            line_style: None,
            gap_color: None,
            alpha: None,
            marker: None,
        }
    }
}

