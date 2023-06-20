use crate::{Point, Bounds, Canvas};

pub enum Clip {
    None,
    Bounds(Point, Point),
}

impl From<&Bounds<Canvas>> for Clip {
    fn from(value: &Bounds<Canvas>) -> Self {
        Clip::Bounds(
            Point(value.xmin(), value.ymin()),
            Point(value.xmax(), value.ymax()),
        )
    }
}