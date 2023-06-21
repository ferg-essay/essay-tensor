use std::str::FromStr;

use essay_plot_base::{Path, Angle, PathCode, Point, Canvas, path_opt::StyleErr};
use essay_tensor::tf32;

use super::paths::{Unit, self};

#[derive(Clone, Debug)]
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
    TickLeft, // #0
    TickRight, // #1
    TickUp, // #2
    TickDown, // #3
    CaretLeft, // #4
    CaretRight, // #5
    CaretUp, // #6
    CaretDown, // #7
    CaretLeftBase, // #8
    CaretRightBase, // #9
    CaretUpBase, // #10
    CaretDownBase, // #11

    Vertices(Vec<[f32; 2]>),
    Path(Path<Unit>),
    Polygon(usize, Angle),
    PolyStar(usize, Angle),
    Asterisk(usize, Angle),
}

impl Markers {
    pub fn is_filled(&self) -> bool {
        match self {
            Self::Square => true,
            _ => false,
        }
    }

    pub fn get_scaled_path(&self, scale: f32) -> Path<Canvas> {
        match self {
            Self::Pixel => {
                self.get_path().scale(2., 2.)
            }
            _ => {
                self.get_path().scale(scale, scale)
            }
        }
    }

    pub fn get_path(&self) -> Path<Unit> {
        match self {
            Self::Circle => paths::circle(),
            Self::Point => paths::circle().scale(0.5, 0.5),
            Self::Pixel => paths::square(), // seems to be identical to square
            Self::TriangleDown => triangle_path().rotate_deg(180.),
            Self::TriangleUp => triangle_path(),
            Self::TriangleLeft => triangle_path().rotate_deg(90.),
            Self::TriangleRight => triangle_path().rotate_deg(270.),
            Self::TriDown => tri_path().rotate_deg(180.),
            Self::TriUp => tri_path(),
            Self::TriLeft => tri_path().rotate_deg(90.),
            Self::TriRight => tri_path().rotate_deg(270.),
            Self::Square => paths::square(),
            Self::Pentagon => paths::unit_polygon(5),
            Self::Hexagon => paths::unit_polygon(6),
            Self::Hexagon2 => paths::unit_polygon(6).rotate_deg(30.),
            Self::Octagon => paths::unit_polygon_alt(8),
            Self::Diamond => paths::unit_polygon(4),
            Self::ThinDiamond => paths::unit_polygon(4).scale(0.5, 1.),
            Self::Star => paths::unit_star(5, 0.381966),
            Self::Plus => plus_path(),
            Self::PlusFilled => plus_filled_path(),
            Self::X => plus_path().rotate_deg(45.),
            Self::XFilled => plus_filled_path().rotate_deg(45.),

            Self::VertLine => vert_path(),
            Self::HorizLine => horiz_path(),

            Self::TickLeft => tick_path().rotate_deg(90.),
            Self::TickRight => tick_path().rotate_deg(270.),
            Self::TickUp => tick_path(),
            Self::TickDown => tick_path().rotate_deg(180.),
            Self::CaretLeft => caret_path().rotate_deg(90.),
            Self::CaretRight => caret_path().rotate_deg(270.),
            Self::CaretUp => caret_path(),
            Self::CaretDown => caret_path().rotate_deg(180.),
            Self::CaretLeftBase => caret_base_path().rotate_deg(90.),
            Self::CaretRightBase => caret_base_path().rotate_deg(270.),
            Self::CaretUpBase => caret_base_path(),
            Self::CaretDownBase => caret_base_path().rotate_deg(180.),

            Self::Path(path) => path.clone(),
            Self::Polygon(n, angle) => {
                paths::unit_polygon(*n).rotate(angle.to_radians())
            }
            Self::PolyStar(n, angle) => {
                paths::unit_star(*n, 0.5).rotate(angle.to_radians())
            }
            Self::Asterisk(n, angle) => {
                paths::unit_asterisk(*n).rotate(angle.to_radians())
            }
            _ => todo!(),
        }
    }
}

impl From<&str> for Markers {
    fn from(value: &str) -> Self {
        Self::from_str(value).unwrap()
    }
}

impl FromStr for Markers {
    type Err = StyleErr;
    
    fn from_str(value: &str) -> Result<Markers, Self::Err> {
        let marker = match value {
            "o" => Self::Circle,
            "." => Self::Point,
            "," => Self::Pixel,
            "v" => Self::TriangleDown,
            "^" => Self::TriangleUp,
            "<" => Self::TriangleLeft,
            ">" => Self::TriangleRight,
            "1" => Self::TriDown,
            "2" => Self::TriUp,
            "3" => Self::TriLeft,
            "4" => Self::TriRight,
            "8" => Self::Octagon,
            "s" => Self::Square,
            "h" => Self::Hexagon,
            "H" => Self::Hexagon2,
            "p" => Self::Pentagon,
            "P" => Self::PlusFilled,
            "+" => Self::Plus,
            "*" => Self::Star,
            "x" => Self::X,
            "X" => Self::XFilled,
            "d" => Self::ThinDiamond,
            "D" => Self::Diamond,
            "|" => Self::VertLine,
            "_" => Self::HorizLine,

            "#0" => Self::TickLeft,
            "#1" => Self::TickRight,
            "#2" => Self::TickUp,
            "#3" => Self::TickDown,
            "#4" => Self::CaretLeft,
            "#5" => Self::CaretRight,
            "#6" => Self::CaretUp,
            "#7" => Self::CaretDown,
            "#8" => Self::CaretLeftBase,
            "#9" => Self::CaretRightBase,
            "#10" => Self::CaretUpBase,
            "#11" => Self::CaretDownBase,

            "" => Self::None,
            "none" => Self::None,

            _ => { return Err(StyleErr(format!("'{}' is an unknown marker symbol", value))); }
        };

        Ok(marker)
    }
}
// filled_markers = '.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*',
// 'h', 'H', 'D', 'd', 'P', 'X'

pub enum FillStyle {
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

fn tri_path() -> Path<Unit> {
    Path::new(vec![
        PathCode::MoveTo(Point(0., 1.)),
        PathCode::LineTo(Point(0., 0.)),
        PathCode::MoveTo(Point(-0.86, -0.5)),
        PathCode::LineTo(Point(0., 0.)),
        PathCode::MoveTo(Point(0.86, -0.5)),
        PathCode::LineTo(Point(0., 0.)),
    ])
}

fn plus_path() -> Path<Unit> {
    Path::new(vec![
        PathCode::MoveTo(Point(-1., 0.)),
        PathCode::LineTo(Point(1., 0.)),
        PathCode::MoveTo(Point(0., -1.)),
        PathCode::LineTo(Point(0., 1.)),
    ])
}

fn plus_filled_path() -> Path<Unit> {
    Path::closed_poly(tf32!([
        [-3., -1.],
        [-3., 1.],
        [-1., 1.],
        [-1., 3.],
        [1., 3.],
        [1., 1.],
        [3., 1.],
        [3., -1.],
        [1., -1.],
        [1., -3.],
        [-1., -3.],
        [-1., -1.],
    ]) / 3.)
}

fn tick_path() -> Path<Unit> {
    Path::new(vec![
        PathCode::MoveTo(Point(0., 0.)),
        PathCode::LineTo(Point(0., 1.)),
    ])
}

fn vert_path() -> Path<Unit> {
    Path::new(vec![
        PathCode::MoveTo(Point(0., -1.)),
        PathCode::LineTo(Point(0., 1.)),
    ])
}

fn horiz_path() -> Path<Unit> {
    Path::new(vec![
        PathCode::MoveTo(Point(-1., 0.)),
        PathCode::LineTo(Point(1., 0.)),
    ])
}

fn caret_path() -> Path<Unit> {
    Path::closed_poly(tf32!([
        [0., 0.], [-0.86, 1.], [0.86, -1.]
    ]))
}

fn caret_base_path() -> Path<Unit> {
    Path::closed_poly(tf32!([
        [-0.86, 0.], [0.86, 0.], [0., 1.]
    ]))
}
