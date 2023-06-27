use essay_tensor::{Tensor};

use super::tricontour::VertId;

pub struct Tile {
    pub(crate) id: TileId,

    pub(crate) verts: [VertId; 3],

    pub(crate) ab: CrossEdge,
    pub(crate) ab_t: f32,
    pub(crate) ab_dual: TileId,

    pub(crate) bc: CrossEdge,
    pub(crate) bc_t: f32,
    pub(crate) bc_dual: TileId,

    pub(crate) ca: CrossEdge,
    pub(crate) ca_t: f32,
    pub(crate) ca_dual: TileId,
}

impl Tile {
    pub fn new(id: TileId, verts: [VertId; 3]) -> Self {
        Self {
            id,
            verts,

            ab: CrossEdge::None,
            ab_t: 0.,
            ab_dual: TileId::none(),

            bc: CrossEdge::None,
            bc_t: 0.,
            bc_dual: TileId::none(),

            ca: CrossEdge::None,
            ca_t: 0.,
            ca_dual: TileId::none(),
        }
    }

    pub(crate) fn cross_ab(&mut self, a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
        let v = match self.ab {
            CrossEdge::Down => {
                self.ab = CrossEdge::None;
                interpolate_vert(self.ab_t, a, b)
            }
            CrossEdge::Up => {
                self.ab = CrossEdge::None;
                interpolate_vert(self.ab_t, a, b)
            }
            _ => panic!("cross_ab with unexpected edge {:?}", self.ab)
        };

        v
    }

    pub(crate) fn cross_bc(&mut self, b: [f32; 2], c: [f32; 2]) -> [f32; 2] {
        match self.bc {
            CrossEdge::Down => {
                self.bc = CrossEdge::None;
                interpolate_vert(self.bc_t, b, c)
            }
            CrossEdge::Up => {
                self.bc = CrossEdge::None;
                interpolate_vert(self.bc_t, b, c)
            }
            _ => panic!("cross_bc with unexpected edge {:?}", self.bc)
        }
    }

    pub(crate) fn cross_ca(&mut self, c: [f32; 2], a: [f32; 2]) -> [f32; 2] {
        match self.ca {
            CrossEdge::Down => {
                self.ca = CrossEdge::None;
                interpolate_vert(self.ca_t, c, a)
            }
            CrossEdge::Up => {
                self.ca = CrossEdge::None;
                interpolate_vert(self.ca_t, c, a)
            }
            _ => panic!("cross_ca with unexpected edge {:?}", self.ca)
        }
    }

    pub(crate) fn clear_from_dual(&mut self, v0: VertId, v1: VertId) {
        if self.verts[0] == v0 && self.verts[1] == v1 {
            self.ab = CrossEdge::None;
        } else if self.verts[1] == v0 && self.verts[2] == v1 {
            self.bc = CrossEdge::None;
        } else if self.verts[2] == v0 && self.verts[0] == v1 {
            self.ca = CrossEdge::None;
        } else {
            panic!("Unexpected clear from dual");
        }
    }

    pub fn init_threshold(&mut self, threshold: f32, z: &Tensor) {
        let a = z[self.verts[0].i()];
        let b = z[self.verts[1].i()];
        let c = z[self.verts[2].i()];

        let (ab, t) = CrossEdge::from_z(threshold, a, b);
        self.ab = ab;
        self.ab_t = t;

        let (bc, t) = CrossEdge::from_z(threshold, b, c);
        self.bc = bc;
        self.bc_t = t;

        let (ca, t) = CrossEdge::from_z(threshold, c, a);
        self.ca = ca;
        self.ca_t = t;
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

    Down, // crossing on the edge, start above to end below
    Up, // crossing on the edge, start below to end above
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

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct TileId(pub(crate) usize);

impl TileId {
    pub const NONE: usize = usize::MAX;

    #[inline]
    pub fn i(&self) -> usize {
        self.0
    }
    
    #[inline]
    pub fn none() -> TileId {
        TileId(Self::NONE)
    }
    
    #[inline]
    pub fn is_none(&self) -> bool {
        self.0 == Self::NONE
    }
}

fn interpolate(threshold: f32, low: f32, high: f32) -> f32 {
    (threshold - low) / (high - low).max(f32::EPSILON)
}

#[inline]
fn interpolate_vert(t: f32, v0: [f32; 2], v1: [f32; 2]) -> [f32; 2] {
    [(1. - t) * v0[0] + t * v1[0], (1. - t) * v0[1] + t * v1[1]]
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