use std::ops::{Index, IndexMut};

pub struct TileGrid {
    tiles: Vec<Tile>,
    x: usize,
    _y: usize
}

impl TileGrid {
    pub fn new(x: usize, y: usize) -> Self {
        assert!(x > 0 && y > 0);

        let mut tiles = Vec::<Tile>::new();

        for _ in 0..y {
            for _ in 0..x {
                tiles.push(Tile::new());
            }
        }

        Self {
            x,
            _y: y,
            tiles,
        }
    }

    pub fn clear(&mut self) {
        for tile in &mut self.tiles {
            tile.clear();
        }
    }
}

impl Index<(usize, usize)> for TileGrid {
    type Output = Tile;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (x, y) = index;

        &self.tiles[y * self.x + x]
    }
}

impl IndexMut<(usize, usize)> for TileGrid {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (x, y) = index;

        &mut self.tiles[y * self.x + x]
    }
}

pub struct Tile {
    pub(crate) left: CrossEdge,
    pub(crate) left_t: f32,

    pub(crate) top: CrossEdge,
    pub(crate) top_t: f32,

    pub(crate) bottom: CrossEdge,
    pub(crate) bottom_t: f32,

    pub(crate) right: CrossEdge,
    pub(crate) right_t: f32,
}

impl Tile {
    pub fn new() -> Self {
        Self {
            left: CrossEdge::Below,
            left_t: 0.,

            top: CrossEdge::Below,
            top_t: 0.,

            bottom: CrossEdge::Below,
            bottom_t: 0.,

            right: CrossEdge::Below,
            right_t: 0.,
        }
    }

    #[inline]
    pub fn left(&mut self, edge: CrossEdge, t: f32) {
        self.left = edge;        
        self.left_t = t;
    }

    pub(crate) fn get_left(&self) -> CrossEdge {
        self.left
    }

    pub(crate) fn cross_left(&mut self) -> Option<f32> {
        match self.left {
            CrossEdge::Down => {
                self.left = CrossEdge::None;
                Some(self.left_t)
            }
            CrossEdge::Up => {
                self.left = CrossEdge::None;
                Some(self.left_t)
            }
            _ => None,
        }
    }

    #[inline]
    pub fn top(&mut self, edge: CrossEdge, t: f32) {
        self.top = edge;        
        self.top_t = t;
    }

    pub(crate) fn get_top(&self) -> CrossEdge {
        self.top
    }

    pub(crate) fn cross_top(&mut self) -> Option<f32> {
        match self.top {
            CrossEdge::Down => {
                self.top = CrossEdge::None;
                Some(self.top_t)
            }
            CrossEdge::Up => {
                self.top = CrossEdge::None;
                Some(self.top_t)
            }
            _ => None,
        }
    }

    pub fn bottom(&mut self, edge: CrossEdge, t: f32) {
        self.bottom = edge;        
        self.bottom_t = t;
    }

    pub(crate) fn get_bottom(&self) -> CrossEdge {
        self.bottom
    }

    pub(crate) fn cross_bottom(&mut self) -> Option<f32> {
        match self.bottom {
            CrossEdge::Down => {
                self.bottom = CrossEdge::None;
                Some(self.bottom_t)
            }
            CrossEdge::Up => {
                self.bottom = CrossEdge::None;
                Some(self.bottom_t)
            }
            _ => None,
        }
    }

    pub fn right(&mut self, edge: CrossEdge, t: f32) {
        self.right = edge;        
        self.right_t = t;
    }

    #[cfg(test)]
    pub(crate) fn get_right(&self) -> CrossEdge {
        self.right
    }

    pub(crate) fn cross_right(&mut self) -> Option<f32> {
        match self.right {
            CrossEdge::Down => {
                self.right = CrossEdge::None;
                Some(self.right_t)
            }
            CrossEdge::Up => {
                self.right = CrossEdge::None;
                Some(self.right_t)
            }
            _ => None,
        }
    }

    pub fn clear(&mut self) {
        self.left = CrossEdge::Below;
        self.top = CrossEdge::Below;
        self.bottom = CrossEdge::Below;
        self.right = CrossEdge::Below;
    }
}

/// directional crossing on a directional edge
/// 
/// Direction defined as measured increasing x or increasing y (w -> e, s -> n).
/// 
/// An equal value is represented by Above to reduce the needed values.
/// 
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum CrossEdge {
    None,

    Below, // all points below the threshold
    Above, // all points above the threshold
    // All, // all points match the threshold

    // StartLow, // initial point is at threshold, other point is higher
    // StartHigh, // initial point is at threshold, other point is lower
    Down, // crossing on the edge, start above to end below
    Up, // crossing on the edge, start below to end above
    // EndLow,
    // EndHigh,
}
impl CrossEdge {
    pub(crate) fn from_z(threshold: f32, z0: f32, z1: f32) -> (Self, f32) {
        if threshold < z0 {
            if threshold < z1 {
                (CrossEdge::Above, 0.)
            } else if z1 < threshold {
                (CrossEdge::Down, 1. - interpolate(threshold, z1, z0))
            } else {
                (CrossEdge::Above, 0.) // equal represented by Above
            }
        } else if z0 < threshold {
            if threshold < z1 {
                (CrossEdge::Up, interpolate(threshold, z0, z1))
            } else if z1 < threshold {
                (CrossEdge::Below, 0.)
            } else {
                (CrossEdge::Up, 1.0) // equal represented by above
            }
        } else { // equal represented by above
            if threshold < z1 {
                (CrossEdge::Above, 0.)
            } else if z1 < threshold {
                (CrossEdge::Down, 0.)
            } else {
                (CrossEdge::Above, 0.)
            }
        }
    }

    #[inline]
    pub(crate) fn is_cross(&self) -> bool {
        match self {
            CrossEdge::Down => true,
            CrossEdge::Up => true,
            _ => false,
        }
    }
}

fn interpolate(threshold: f32, low: f32, high: f32) -> f32 {
    (threshold - low) / (high - low).max(f32::EPSILON)
}

#[cfg(test)]
mod test {
    use crate::contour::tile::CrossEdge;

    #[test]
    fn test_edge() {
        assert_eq!(CrossEdge::from_z(0., 0., 0.), (CrossEdge::Above, 0.));

        assert_eq!(CrossEdge::from_z(2., 0., 1.), (CrossEdge::Below, 0.));
        assert_eq!(CrossEdge::from_z(-2., 0., 1.), (CrossEdge::Above, 0.));

        assert_eq!(CrossEdge::from_z(0., 0., 1.), (CrossEdge::Above, 0.));
        assert_eq!(CrossEdge::from_z(0.5, 0., 1.), (CrossEdge::Up, 0.5));
        assert_eq!(CrossEdge::from_z(1., 0., 1.), (CrossEdge::Up, 0.5));

        assert_eq!(CrossEdge::from_z(2., 1., 0.), (CrossEdge::Below, 0.));
        assert_eq!(CrossEdge::from_z(-2., 1., 0.), (CrossEdge::Above, 0.));

        assert_eq!(CrossEdge::from_z(0., 1., 0.), (CrossEdge::Above, 0.));
        assert_eq!(CrossEdge::from_z(0.5, 1., 0.), (CrossEdge::Down, 0.5));
        assert_eq!(CrossEdge::from_z(1., 1., 0.), (CrossEdge::Down, 0.));
    }

    #[test]
    fn edge_interpolate() {
        assert_eq!(CrossEdge::from_z(0.5, 0., 1.), (CrossEdge::Up, 0.5));

        assert_eq!(CrossEdge::from_z(0.25, 0., 1.), (CrossEdge::Up, 0.25));
        assert_eq!(CrossEdge::from_z(0.75, 0., 1.), (CrossEdge::Up, 0.75));

        assert_eq!(CrossEdge::from_z(0.25, 1., 0.), (CrossEdge::Down, 0.75));
        assert_eq!(CrossEdge::from_z(0.75, 1., 0.), (CrossEdge::Down, 0.25));
    }
}