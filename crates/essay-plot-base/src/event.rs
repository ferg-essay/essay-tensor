use crate::Point;

// TODO: Consider changing these to abstract events like Pan, Zoom because
// of tablets, etc.
#[derive(Clone, Debug)]
pub enum CanvasEvent {
    MouseLeftPress(Point),
    MouseLeftRelease(Point),
    Pan(Point, Point, Point),
    ResetView(Point),

    MouseRightPress(Point),
    MouseRightRelease(Point),
    MouseRightDrag(Point, Point),
    ZoomBounds(Point, Point),
    MouseRightDoubleClick(Point),

    MouseMiddlePress(Point),
    MouseMiddleRelease(Point),
    MouseMiddleDrag(Point, Point),
    MouseMiddleDoubleClick(Point),
}

impl CanvasEvent {
    #[inline]
    pub fn point(&self) -> Point {
        match self {
            CanvasEvent::MouseLeftPress(point) => *point,
            CanvasEvent::MouseLeftRelease(point) => *point,
            CanvasEvent::Pan(point, _, _) => *point,
            CanvasEvent::ResetView(point) => *point,

            CanvasEvent::MouseRightPress(point) => *point,
            CanvasEvent::MouseRightRelease(point) => *point,
            CanvasEvent::MouseRightDrag(point, _) => *point,
            CanvasEvent::ZoomBounds(point, _) => *point,
            CanvasEvent::MouseRightDoubleClick(point) => *point,

            CanvasEvent::MouseMiddlePress(point) => *point,
            CanvasEvent::MouseMiddleRelease(point) => *point,
            CanvasEvent::MouseMiddleDrag(point, _) => *point,
            CanvasEvent::MouseMiddleDoubleClick(point) => *point,
        }
    }
}