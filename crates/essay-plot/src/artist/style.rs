use essay_plot_base::{Color, JoinStyle, CapStyle, PathOpt, LineStyle};


#[derive(Clone)]
pub struct Style {
    color: Option<Color>,
    facecolor: Option<Color>,
    edgecolor: Option<Color>,

    linewidth: Option<f32>,
    joinstyle: Option<JoinStyle>,
    capstyle: Option<CapStyle>,

    linestyle: Option<LineStyle>,
    gapcolor: Option<Color>,
}

impl Style {
    pub fn new() -> Style {
        Style::default()
    }

    pub fn color(&mut self, color: impl Into<Color>) -> &mut Self {
        // TODO: is color a default or an assignment?
        self.color = Some(color.into());

        self
    }

    pub fn facecolor(&mut self, color: impl Into<Color>) -> &mut Self {
        self.facecolor = Some(color.into());

        self
    }

    pub fn edgecolor(&mut self, color: impl Into<Color>) -> &mut Self {
        self.edgecolor = Some(color.into());

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

    pub fn capstyle(&mut self, capstyle: impl Into<CapStyle>) -> &mut Self {
        self.capstyle = Some(capstyle.into());

        self
    }
    /*
    pub fn chain<'a>(prev: &'a dyn StyleOpt, next: &'a dyn StyleOpt) -> StyleChain<'a>
    {
        StyleChain::new(prev, next)
    }
    */
}

impl PathOpt for Style {
    fn get_fill_color(&self) -> &Option<Color> {
        match &self.facecolor {
            Some(_color) => &self.facecolor,
            None => &self.color,
        }
    }

    fn get_line_color(&self) -> &Option<Color> {
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

impl Default for Style {
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

struct PropCycle {
    fill_colors: Option<Vec<Color>>,
    edge_colors: Option<Vec<Color>>,
    line_widths: Option<Vec<f32>>,
    line_styles: Option<LineStyle>,
}
/*
pub struct StyleChain<'a> {
    prev: &'a dyn StyleOpt,
    next: &'a dyn StyleOpt,
}

impl<'a> StyleChain<'a> {
    pub fn new(prev: &'a dyn StyleOpt, next: &'a dyn StyleOpt) -> Self {
        Self {
            prev,
            next,
        }
    }
}
impl StyleOpt for StyleChain<'_> {
    fn get_fill_color(&self) -> &Option<Color> {
        match self.next.get_fill_color() {
            Some(_) => self.next.get_fill_color(),
            None => self.prev.get_fill_color()
        }
    }

    fn get_line_color(&self) -> &Option<Color> {
        match self.next.get_line_color() {
            Some(_) => self.next.get_line_color(),
            None => self.prev.get_line_color()
        }
    }

    fn get_line_width(&self) -> &Option<f32> {
        match self.next.get_line_width() {
            Some(_) => self.next.get_line_width(),
            None => self.prev.get_line_width()
        }
    }

    fn get_join_style(&self) -> &Option<JoinStyle> {
        match self.next.get_join_style() {
            Some(_) => self.next.get_join_style(),
            None => self.prev.get_join_style()
        }
    }

    fn get_cap_style(&self) -> &Option<CapStyle> {
        match self.next.get_cap_style() {
            Some(_) => self.next.get_cap_style(),
            None => self.prev.get_cap_style()
        }
    }
}
*/
