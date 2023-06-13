use crate::Point;

// TODO: Consider changing these to abstract events like Pan, Zoom because
// of tablets, etc.
#[derive(Clone, Debug)]
pub enum CanvasEvent {
    MouseLeftPress(Point),
    MouseLeftRelease(Point),
    MouseLeftDrag(Point, Point, Point),
    MouseLeftDoubleClick(Point),

    MouseRightPress(Point),
    MouseRightRelease(Point),
    MouseRightDrag(Point, Point),
    MouseRightDrop(Point, Point),
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
            CanvasEvent::MouseLeftDrag(point, _, _) => *point,
            CanvasEvent::MouseLeftDoubleClick(point) => *point,

            CanvasEvent::MouseRightPress(point) => *point,
            CanvasEvent::MouseRightRelease(point) => *point,
            CanvasEvent::MouseRightDrag(point, _) => *point,
            CanvasEvent::MouseRightDrop(point, _) => *point,
            CanvasEvent::MouseRightDoubleClick(point) => *point,

            CanvasEvent::MouseMiddlePress(point) => *point,
            CanvasEvent::MouseMiddleRelease(point) => *point,
            CanvasEvent::MouseMiddleDrag(point, _) => *point,
            CanvasEvent::MouseMiddleDoubleClick(point) => *point,
        }
    }
}