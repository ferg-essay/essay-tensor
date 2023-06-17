use essay_plot_base::{Color, JoinStyle, CapStyle, PathOpt, LineStyle};


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

    pub fn facecolor(&mut self, color: impl Into<Color>) -> &mut Self {
        self.facecolor = Some(color.into());

        self
    }

    pub fn edgecolor(&mut self, color: impl Into<Color>) -> &mut Self {
        self.edgecolor = Some(color.into());

        self
    }

    pub fn line_style(&mut self, line_style: impl Into<LineStyle>) -> &mut Self {
        self.linestyle = Some(line_style.into());

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
}

impl PathOpt for PathStyle {
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

struct PropCycle {
    fill_colors: Option<Vec<Color>>,
    edge_colors: Option<Vec<Color>>,
    line_widths: Option<Vec<f32>>,
    line_styles: Option<LineStyle>,
}
