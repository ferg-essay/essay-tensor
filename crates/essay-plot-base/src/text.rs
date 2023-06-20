
#[derive(Clone, Copy, Debug)]
pub struct TextStyle {
    size: Option<f32>,
    vert_align: Option<VertAlign>,
    horiz_align: Option<HorizAlign>,
}

#[derive(Clone, Copy, Debug)]
pub enum VertAlign {
    Bottom,
    Center,
    Top,
}

#[derive(Clone, Copy, Debug)]
pub enum HorizAlign {
    Left,
    Center,
    Right,
}

impl TextStyle {
    pub const SIZE_DEFAULT : f32 = 12.;
    pub const HALIGN_DEFAULT : HorizAlign = HorizAlign::Center;
    pub const VALIGN_DEFAULT : VertAlign = VertAlign::Bottom;

    pub fn new() -> Self {
        Self {
            size: None,
            vert_align: None,
            horiz_align: None,
        }
    }

    #[inline]
    pub fn get_size(&self) -> &Option<f32> {
        &self.size
    }

    pub fn size(&mut self, size: f32) -> &mut Self {
        self.size = Some(size);

        self
    }

    #[inline]
    pub fn get_height_align(&self) -> &Option<VertAlign> {
        &self.vert_align
    }

    pub fn valign(&mut self, align: VertAlign) {
        self.vert_align = Some(align);
    }

    #[inline]
    pub fn get_width_align(&self) -> &Option<HorizAlign> {
        &self.horiz_align
    }

    pub fn width_align(&mut self, align: HorizAlign) {
        self.horiz_align = Some(align);
    }

}