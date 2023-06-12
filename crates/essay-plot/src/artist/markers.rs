use essay_tensor::tf32;

use crate::graph::{Unit, Point};

use super::{Path, Angle, PathCode};

pub enum Markers {
    None,
    Point, // '.'
    Pixel, // ','
    Circle, // 'o'
    TriangleDown, // 'v'
    TriangleUp, // '^'
    TriangleLeft, // '<'
    TriangleRight, // '>'
    TriDown, // '1'
    TriUp, // '2'
    TriLeft, // '3'
    TriRight, // '4'
    Octagon, // '8'
    Square, // 's'
    Pentagon, // 'p'
    PlusFilled, // 'P'
    Star, // '*'
    Hexagon, // 'h'
    Hexagon2, // 'H'
    Plus, // '+'
    X, // 'x'
    XFilled, // 'X'
    Diamond, // 'D'
    ThinDiamond, // 'd' 
    VertLine, // '|'
    HorizLine, // '_'
    TickLeft, // `0
    TickRight, // `1
    TickUp, // `2
    TickDown, // `3
    CaretLeft, // `4
    CaretRight, // `5
    CaretUp, // `6
    CaretDown, // `7
    CaretLeftBase, // `8
    CaretRightBase, // `9
    CaretUpBase, // `10
    CaretDownBase, // `11

    Vertices(Vec<[f32; 2]>),
    Path(Path<Unit>),
    Polygon(usize, Angle),
    PolyStar(usize, Angle),
    Asterisk(usize, Angle,)
}

impl Markers {
    pub fn is_filled(&self) -> bool {
        match self {
            Square => true,
            _ => false,
        }
    }

    pub fn get_path(&self) -> Path<Unit> {
        match self {
            Square => Path::unit(),
            TriangleUp => triangle_path(),
            Self::Path(path) => path.clone(),
            _ => todo!(),
        }
    }
}
// filled_markers = '.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*',
// 'h', 'H', 'D', 'd', 'P', 'X'

enum FillStyle {
    None,
    Left,
    Right,
    Bottom,
    Top,
    Full,
}

fn triangle_path() -> Path<Unit> {
    Path::closed_poly(tf32!([
        [0., 1.], [-1., -1.], [1., -1.]
    ]))
}

fn triangle_path_up() -> Path<Unit> {
    Path::closed_poly(tf32!([
        [0., 1.], [-3./5., -1./5.], [3./5., -1./5.],
    ]))
}

fn triangle_path_down() -> Path<Unit> {
    Path::closed_poly(tf32!([
        [-3./5., -1./5.], [3./5., -1./5.], [1., -1.], [-1., -1.]
    ]))
}

fn triangle_path_left() -> Path<Unit> {
    Path::closed_poly(tf32!([
        [0., 1.], [0., -1.], [-1., -1.],
    ]))
}

fn triangle_path_right() -> Path<Unit> {
    Path::closed_poly(tf32!([
        [0., 1.], [0., -1.], [1., -1.],
    ]))
}

fn plus_path() -> Path<Unit> {
    Path::new(vec![
        PathCode::MoveTo(Point(-1., 0.)),
        PathCode::LineTo(Point(1., 0.)),
        PathCode::MoveTo(Point(0., -1.)),
        PathCode::LineTo(Point(0., 1.)),
    ])
}
