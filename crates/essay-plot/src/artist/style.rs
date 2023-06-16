use essay_plot_base::{Color, JoinStyle, CapStyle, StyleOpt};


#[derive(Clone)]
pub struct Style {
    color: Option<Color>,
    facecolor: Option<Color>,
    edgecolor: Option<Color>,

    linewidth: Option<f32>,
    joinstyle: Option<JoinStyle>,
    capstyle: Option<CapStyle>,
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

    pub fn chain<'a>(prev: &'a dyn StyleOpt, next: &'a dyn StyleOpt) -> StyleChain<'a>
    {
        StyleChain::new(prev, next)
    }
}

impl StyleOpt for Style {
    fn get_facecolor(&self) -> &Option<Color> {
        match &self.facecolor {
            Some(color) => &self.facecolor,
            None => &self.color,
        }
    }

    fn get_edgecolor(&self) -> &Option<Color> {
        match &self.edgecolor {
            Some(color) => &self.edgecolor,
            None => &self.color,
        }
    }

    fn get_linewidth(&self) -> &Option<f32> {
        &self.linewidth
    }

    fn get_joinstyle(&self) -> &Option<JoinStyle> {
        &self.joinstyle
    }

    fn get_capstyle(&self) -> &Option<CapStyle> {
        &self.capstyle
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
        }
    }
}

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
    fn get_facecolor(&self) -> &Option<Color> {
        match self.next.get_facecolor() {
            Some(_) => self.next.get_facecolor(),
            None => self.prev.get_facecolor()
        }
    }

    fn get_edgecolor(&self) -> &Option<Color> {
        match self.next.get_edgecolor() {
            Some(_) => self.next.get_edgecolor(),
            None => self.prev.get_edgecolor()
        }
    }

    fn get_linewidth(&self) -> &Option<f32> {
        match self.next.get_linewidth() {
            Some(_) => self.next.get_linewidth(),
            None => self.prev.get_linewidth()
        }
    }

    fn get_joinstyle(&self) -> &Option<JoinStyle> {
        match self.next.get_joinstyle() {
            Some(_) => self.next.get_joinstyle(),
            None => self.prev.get_joinstyle()
        }
    }

    fn get_capstyle(&self) -> &Option<CapStyle> {
        match self.next.get_capstyle() {
            Some(_) => self.next.get_capstyle(),
            None => self.prev.get_capstyle()
        }
    }
}
