
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point(pub f32, pub f32);

impl Point {
    #[inline]
    pub fn x(&self) -> f32 {
        self.0
    }

    #[inline]
    pub fn y(&self) -> f32 {
        self.1
    }

    #[inline]
    pub fn is_below(&self, p0: &Point, p1: &Point) -> bool {
        let Point(x, y) = self;
        let Point(x0, y0) = p0;
        let Point(x1, y1) = p1;

        if x0 == x1 {
            false
        } else if x0 <= x && x < x1 || x1 < x && x <= x0 {
            let y_line = (y0 * (x1 - x) + y1 * (x - x0)) / (x1 - x0);

            *y < y_line
        } else {
            false
        }
    }

    #[inline]
    pub fn dist(&self, p: &Point) -> f32 {
        let dx = self.0 - p.0;
        let dy = self.1 - p.1;

        dx.hypot(dy)
    }
}

impl From<[f32; 2]> for Point {
    #[inline]
    fn from(value: [f32; 2]) -> Self {
        Point(value[0], value[1])
    }
}

impl From<&[f32; 2]> for Point {
    #[inline]
    fn from(value: &[f32; 2]) -> Self {
        Point(value[0], value[1])
    }
}

// angle in [0., 1.]
#[derive(Clone, Copy, Debug)]
pub enum Angle {
    Rad(f32),
    Deg(f32),
    Unit(f32),
}

impl Angle {
    pub fn to_radians(&self) -> f32 {
        match self {
            Angle::Rad(rad) => *rad,
            Angle::Deg(deg) => deg.to_radians(),
            Angle::Unit(unit) => (unit * 360.).to_radians(),
        }
    }

    pub fn to_degrees(&self) -> f32 {
        match self {
            Angle::Rad(rad) => rad.to_degrees(),
            Angle::Deg(deg) => *deg,
            Angle::Unit(unit) => unit * 360.,
        }
    }

    pub fn to_unit(&self) -> f32 {
        match self {
            Angle::Rad(rad) => rad.to_degrees() / 360.,
            Angle::Deg(deg) => deg / 360.,
            Angle::Unit(unit) => *unit,
        }
    }
}

impl From<f32> for Angle {
    fn from(value: f32) -> Self {
        Angle::Rad(value)
    }
}
