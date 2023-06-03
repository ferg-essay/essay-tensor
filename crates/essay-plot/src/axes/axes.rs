use super::rect::Rect;

pub struct Axes {
    data_lim: Rect,
    view_lim: Rect,   
}

impl Axes {
    pub fn new(view: impl Into<Rect>) -> Self {
        Self {
            view_lim: view.into(),
            data_lim: [-1., -1., 1., 1.].into(),
        }
    }
}