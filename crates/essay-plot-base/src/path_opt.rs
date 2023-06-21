use std::str::FromStr;

use super::Color;

///
/// Renderer options for a path, including the fill color, line (edge) color,
/// and width.
/// 
/// Path options are typically organized as a stack to enable defaults.
/// 
pub trait PathOpt {
    ///
    /// Color to fill a closed path and the default line color.
    /// 
    fn get_face_color(&self) -> &Option<Color>;

    ///
    /// Color for a path's line when the outline differs from the fill
    /// color
    /// 
    fn get_edge_color(&self) -> &Option<Color>;

    ///
    /// Dash pattern for the line, defaults to a LineStyle::Solid.
    /// 
    fn get_line_style(&self) -> &Option<LineStyle>;

    ///
    /// Line width in logical pixels (points for physical dimensions).
    /// 
    fn get_line_width(&self) -> &Option<f32>;

    ///
    /// Style to join two line segments, defaults to JoinStyle::Miter.
    /// 
    fn get_join_style(&self) -> &Option<JoinStyle>;

    ///
    /// Style for a line end, defaults to CapStyle::Butt.
    /// 
    fn get_cap_style(&self) -> &Option<CapStyle>;

    ///
    /// Overriding alpha (transparency) for the path. This alpha will be
    /// multiplied with any alpha in the path's color.
    /// 
    fn get_alpha(&self) -> &Option<f32>;

    ///
    /// Texture used to fill a closed path.
    /// 
    fn get_texture(&self) -> &Option<TextureId>;

    ///
    /// Pushes this style on an option stack. Top styles will override
    /// lower items.
    /// 
    fn push<'a>(&'a self, prev: &'a dyn PathOpt)-> Stack<'a>
    where
        Self: Sized
    {
        Stack::new(prev, self)
    }
    
    // hatch (texture)
    // clip
    // alpha (forced alpha)

    // antialiased
    // gapcolor
    // linestyle
    // dash_cap_style
    // solid_cap_style
}

///
/// Renderer-specific texture id used to fill paths. The TextureId is
/// obtained from the renderer when creating the texture, and passed back
/// at draw time.
/// 
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TextureId(usize);

impl TextureId {
    pub fn new(index: usize) -> Self {
        Self(index)
    }

    #[inline]
    pub fn index(&self) -> usize {
        self.0
    }
}

pub struct Stack<'a> {
    prev: &'a dyn PathOpt,
    next: &'a dyn PathOpt,
}

impl<'a> Stack<'a> {
    pub fn new(prev: &'a dyn PathOpt, next: &'a dyn PathOpt) -> Self {
        Self {
            prev,
            next,
        }
    }
}

impl PathOpt for Stack<'_> {
    fn get_face_color(&self) -> &Option<Color> {
        match self.next.get_face_color() {
            Some(_) => self.next.get_face_color(),
            None => self.prev.get_face_color(),
        }
    }

    fn get_edge_color(&self) -> &Option<Color> {
        match self.next.get_edge_color() {
            Some(_) => self.next.get_edge_color(),
            None => self.prev.get_edge_color(),
        }
    }

    fn get_line_style(&self) -> &Option<LineStyle> {
        match self.next.get_line_style() {
            Some(_) => self.next.get_line_style(),
            None => self.prev.get_line_style(),
        }
    }

    fn get_line_width(&self) -> &Option<f32> {
        match self.next.get_line_width() {
            Some(_) => self.next.get_line_width(),
            None => self.prev.get_line_width(),
        }
    }

    fn get_join_style(&self) -> &Option<JoinStyle> {
        match self.next.get_join_style() {
            Some(_) => self.next.get_join_style(),
            None => self.prev.get_join_style(),
        }
    }

    fn get_cap_style(&self) -> &Option<CapStyle> {
        match self.next.get_cap_style() {
            Some(_) => self.next.get_cap_style(),
            None => self.prev.get_cap_style(),
        }
    }

    fn get_alpha(&self) -> &Option<f32> {
        match self.next.get_alpha() {
            Some(_) => self.next.get_alpha(),
            None => self.prev.get_alpha(),
        }
    }

    fn get_texture(&self) -> &Option<TextureId> {
        match self.next.get_texture() {
            Some(_) => self.next.get_texture(),
            None => self.prev.get_texture(),
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub enum LineStyle {
    None, // draw nothing - distinct from Option::None
    Solid,
    Dashed,
    DashDot,
    Dot,
    OnOff(Vec<f32>),
}

impl LineStyle {
    pub fn to_pattern(&self, lw: f32) -> Vec<f32> {
        match self {
            Self::Dot => { vec![lw, 2. * lw] }
            Self::Dashed => { vec![4. * lw, 2. * lw] }
            Self::DashDot => { vec![4. * lw, 2. * lw, lw, 2. * lw] }
            _ => panic!("Unexpected linestyle {:?}", self)
        }
    }
}

impl From<&LineStyle> for LineStyle {
    fn from(value: &LineStyle) -> Self {
        value.clone()
    }
}

impl From<&str> for LineStyle {
    fn from(name: &str) -> Self {
        match name {
            "" => Self::Solid,
            "-" => Self::Solid,
            ":" => Self::Dot,
            "--" => Self::Dashed,
            "-." => Self::DashDot,
            _ => panic!("'{}' is an unknown line style", name)
        }
    }
}

impl FromStr for LineStyle {
    type Err = StyleErr;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name {
            "" => Ok(Self::Solid),
            "-" => Ok(Self::Solid),
            "solid" => Ok(Self::Solid),
            ":" => Ok(Self::Dot),
            "dot" => Ok(Self::Dot),
            "--" => Ok(Self::Dashed),
            "dashed" => Ok(Self::Dashed),
            "-." => Ok(Self::DashDot),
            "dashdot" => Ok(Self::DashDot),
            "none" => Ok(Self::None),
            _ => Err(StyleErr(format!("'{}' is an unknown line_style", name)))
        }
    }
}

#[derive(Clone, Debug)]
pub struct StyleErr(pub String);


#[derive(Clone, Copy, PartialEq, Debug)]
pub enum JoinStyle {
    Bevel,
    Miter,
    Round,
}

impl FromStr for JoinStyle {
    type Err = StyleErr;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name {
            "bevel" => Ok(Self::Bevel),
            "miter" => Ok(Self::Miter),
            "round" => Ok(Self::Round),
            _ => Err(StyleErr(format!("'{}' is an unknown join_style", name)))
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum CapStyle {
    Butt,
    Round,
    Projecting,
}

impl FromStr for CapStyle {
    type Err = StyleErr;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name {
            "butt" => Ok(Self::Butt),
            "round" => Ok(Self::Round),
            "projecting" => Ok(Self::Projecting),
            _ => Err(StyleErr(format!("'{}' is an unknown cap_style", name)))
        }
    }
}
