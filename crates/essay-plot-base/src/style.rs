use super::Color;

pub trait StyleOpt {
    fn get_color(&self) -> &Option<Color>;

    fn get_linewidth(&self) -> &Option<f32>;
    fn get_joinstyle(&self) -> &Option<JoinStyle>;

    // drawstyle
    // antialiased
    // gapcolor
    // linestyle
    // dash_cap_style
    // solid_cap_style
    // marker
    // marker_edge_color
    // marker_edge_width
    // marker_face_color
    // marker_face_color_alt
    // marker_size
}

#[derive(Clone, PartialEq, Debug)]
pub enum JoinStyle {
    Bevel,
    Mitre,
    Rounded,
}

#[derive(Clone, PartialEq, Debug)]
pub enum CapStyle {
    Rounded,
    Extended,
}

#[derive(Clone, PartialEq, Debug)]
pub enum DrawStyle {
    StepsPre,
    StepsMid,
    StepsPost
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

#[derive(Clone)]
pub struct Style {
    color: Option<Color>,

    linewidth: Option<f32>,
    joinstyle: Option<JoinStyle>,
}

impl Style {
    pub fn new() -> Style {
        Style::default()
    }

    pub fn color(&mut self, color: impl Into<Color>) -> &mut Self {
        self.color = Some(color.into());

        self
    }

    pub fn linewidth(&mut self, linewidth: f32) -> &mut Self {
        assert!(linewidth > 0.);

        self.linewidth = Some(linewidth);

        self
    }

    pub fn joinstyle(&mut self, joinstyle: impl Into<JoinStyle>) -> &mut Self {
        self.joinstyle = Some(joinstyle.into());

        self
    }

    pub fn chain<'a>(prev: &'a dyn StyleOpt, next: &'a dyn StyleOpt) -> Chain<'a>
    {
        Chain::new(prev, next)
    }
}

impl StyleOpt for Style {
    fn get_color(&self) -> &Option<Color> {
        &self.color
    }

    fn get_linewidth(&self) -> &Option<f32> {
        &self.linewidth
    }

    fn get_joinstyle(&self) -> &Option<JoinStyle> {
        &self.joinstyle
    }
}

impl Default for Style {
    fn default() -> Self {
        Self { 
            color: Default::default(), 
            linewidth: Default::default(), 
            joinstyle: Default::default(),
        }
    }
}

pub struct Chain<'a> {
    prev: &'a dyn StyleOpt,
    next: &'a dyn StyleOpt,
}

impl<'a> Chain<'a> {
    pub fn new(prev: &'a dyn StyleOpt, next: &'a dyn StyleOpt) -> Self {
        Self {
            prev,
            next,
        }
    }
}
impl StyleOpt for Chain<'_> {
    fn get_color(&self) -> &Option<Color> {
        match self.next.get_color() {
            Some(color) => self.next.get_color(),
            None => self.prev.get_color()
        }
    }

    fn get_linewidth(&self) -> &Option<f32> {
        match self.next.get_linewidth() {
            Some(linewidth) => self.next.get_linewidth(),
            None => self.prev.get_linewidth()
        }
    }

    fn get_joinstyle(&self) -> &Option<JoinStyle> {
        match self.next.get_joinstyle() {
            Some(joinstyle) => self.next.get_joinstyle(),
            None => self.prev.get_joinstyle()
        }
    }
}
