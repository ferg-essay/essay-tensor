
pub trait Coord: 'static {}

///
/// Display coordinates are in pixels for screens, or points for text rendering.
///
#[derive(Clone, Copy, Debug)]
pub struct Display {}

impl Coord for Display {
}

///
/// Unit coordinates are [0, 1] space, used for markers and similar templates.
/// 
#[derive(Clone, Copy, Debug)]
pub struct Unit {}

impl Coord for Unit {
}
