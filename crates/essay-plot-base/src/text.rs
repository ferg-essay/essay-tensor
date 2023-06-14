
#[derive(Clone, Copy, Debug)]
pub struct TextStyle {
    size: Option<f32>,
    height_align: Option<HeightAlign>,
    width_align: Option<WidthAlign>,
}

#[derive(Clone, Copy, Debug)]
pub enum HeightAlign {
    Bottom,
    Center,
    Top,
}

#[derive(Clone, Copy, Debug)]
pub enum WidthAlign {
    Left,
    Center,
    Right,
}

impl TextStyle {
    pub const SIZE_DEFAULT : f32 = 12.;
    pub const HALIGN_DEFAULT : WidthAlign = WidthAlign::Center;
    pub const VALIGN_DEFAULT : HeightAlign = HeightAlign::Bottom;

    pub fn new() -> Self {
        Self {
            size: None,
            height_align: None,
            width_align: None,
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
    pub fn get_height_align(&self) -> &Option<HeightAlign> {
        &self.height_align
    }

    pub fn height_align(&mut self, align: HeightAlign) {
        self.height_align = Some(align);
    }

    #[inline]
    pub fn get_width_align(&self) -> &Option<WidthAlign> {
        &self.width_align
    }

    pub fn width_align(&mut self, align: WidthAlign) {
        self.width_align = Some(align);
    }

}