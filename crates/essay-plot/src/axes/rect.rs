#[derive(Clone, Debug)]
pub struct Rect {
    left: f32,
    bottom: f32,
    width: f32,
    height: f32,
}

impl Rect {
    pub fn new(left: f32, bottom: f32, width: f32, height: f32) -> Self {
        Self {
            left,
            bottom,
            width,
            height
        }
    }

    #[inline]
    pub fn left(&self) -> f32 {
        self.left
    }

    #[inline]
    pub fn bottom(&self) -> f32 {
        self.bottom
    }

    #[inline]
    pub fn width(&self) -> f32 {
        self.width
    }

    #[inline]
    pub fn height(&self) -> f32 {
        self.height
    }
}

impl From<(f32, f32, f32, f32)> for Rect {
    fn from(value: (f32, f32, f32, f32)) -> Self {
        Rect::new(value.0, value.1, value.2, value.3)
    }
}

impl From<[f32; 4]> for Rect {
    fn from(value: [f32; 4]) -> Self {
        Rect::new(value[0], value[1], value[2], value[3])
    }
}

impl From<Vec<f32>> for Rect {
    fn from(value: Vec<f32>) -> Self {
        assert!(value.len() == 4);

        Rect::new(value[0], value[1], value[2], value[3])
    }
}
