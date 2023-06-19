use essay_plot_base::{Color, JoinStyle, CapStyle, PathOpt, LineStyle, TextureId};


#[derive(Clone)]
pub struct PathStyle {
    color: Option<Color>,
    facecolor: Option<Color>,
    edgecolor: Option<Color>,

    linewidth: Option<f32>,
    joinstyle: Option<JoinStyle>,
    capstyle: Option<CapStyle>,

    linestyle: Option<LineStyle>,
    gapcolor: Option<Color>,
}

impl PathStyle {
    pub fn new() -> PathStyle {
        PathStyle::default()
    }

    pub fn color(&mut self, color: impl Into<Color>) -> &mut Self {
        // TODO: is color a default or an assignment?
        self.color = Some(color.into());

        self
    }

    pub fn fill_color(&mut self, color: impl Into<Color>) -> &mut Self {
        self.facecolor = Some(color.into());

        self
    }

    pub fn edge_color(&mut self, color: impl Into<Color>) -> &mut Self {
        self.edgecolor = Some(color.into());

        self
    }

    pub fn line_style(&mut self, line_style: impl Into<LineStyle>) -> &mut Self {
        self.linestyle = Some(line_style.into());

        self
    }

    pub fn line_width(&mut self, linewidth: f32) -> &mut Self {
        assert!(linewidth > 0.);

        self.linewidth = Some(linewidth);

        self
    }

    pub fn join_style(&mut self, joinstyle: impl Into<JoinStyle>) -> &mut Self {
        self.joinstyle = Some(joinstyle.into());

        self
    }

    pub fn cap_style(&mut self, capstyle: impl Into<CapStyle>) -> &mut Self {
        self.capstyle = Some(capstyle.into());

        self
    }
}

impl PathOpt for PathStyle {
    fn get_fill_color(&self) -> &Option<Color> {
        match &self.facecolor {
            Some(_color) => &self.facecolor,
            None => &self.color,
        }
    }

    fn get_edge_color(&self) -> &Option<Color> {
        match &self.edgecolor {
            Some(_color) => &self.edgecolor,
            None => &self.color,
        }
    }

    fn get_line_width(&self) -> &Option<f32> {
        &self.linewidth
    }

    fn get_join_style(&self) -> &Option<JoinStyle> {
        &self.joinstyle
    }

    fn get_cap_style(&self) -> &Option<CapStyle> {
        &self.capstyle
    }

    fn get_line_style(&self) -> &Option<LineStyle> {
        &self.linestyle
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
            facecolor: Default::default(), 
            edgecolor: Default::default(), 
            linewidth: Default::default(), 
            joinstyle: Default::default(),
            capstyle: Default::default(),
            linestyle: Default::default(),
            gapcolor: Default::default(),
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
    fn get_fill_color(&self) -> &Option<Color> {
        if self.cycle.is_fill_color_set() {
            self.cycle.get_fill_color(self.index)
        } else {
            self.prev.get_fill_color()
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

