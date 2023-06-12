

pub struct TextStyle {
    size: Option<f32>,
}

impl TextStyle {
    pub fn new() -> Self {
        Self {
            size: None,
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
}