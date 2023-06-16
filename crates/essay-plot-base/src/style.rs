use super::Color;

pub trait StyleOpt {
    fn get_facecolor(&self) -> &Option<Color>;
    fn get_edgecolor(&self) -> &Option<Color>;

    fn get_linewidth(&self) -> &Option<f32>;
    fn get_joinstyle(&self) -> &Option<JoinStyle>;
    fn get_capstyle(&self) -> &Option<CapStyle>;

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
    Miter,
    Round,
}

#[derive(Clone, PartialEq, Debug)]
pub enum CapStyle {
    Butt,
    Round,
    Projecting,
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
