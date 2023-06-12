
pub trait CoordMarker {}

///
/// Display coordinates are in pixels for screens, or points for text rendering.
///
#[derive(Clone, Copy, Debug)]
pub struct Display {}

impl CoordMarker for Display {
}

///
/// Unit coordinates are [0, 1] space, used for markers and similar templates.
/// 
#[derive(Clone, Copy, Debug)]
pub struct Unit {}

impl CoordMarker for Unit {
}
