use std::{collections::HashMap, ops::{Index, IndexMut}};

use essay_tensor::{Tensor, tensor::TensorVec};

use crate::{tri::Triangulation};

use super::{tritile::{TileId, Tile, CrossEdge}};

pub struct TriContourGenerator {
    tiles: Tiles,
}

impl TriContourGenerator {
    pub fn new(tri: &Triangulation, z: Tensor) -> Self {
        assert!(z.rank() == 1, "contour z data must be 1D (rank 1) {:?}", z.shape().as_slice());
        assert_eq!(tri.vertices().rows(), z.len(), "contour z length must match vertices (vertices {:?}, z {:?})", 
            tri.vertices().shape().as_slice(), z.shape().as_slice());

        //let tile_cols = z.cols() - 1;
        //let tile_rows = z.rows() - 1;

        let tiles = Tiles::new(tri, z);

        Self {
            tiles,
        }
    }

    pub fn init_threshold(&mut self, threshold: f32) {
        self.tiles.init_threshold(threshold);
    }

    pub fn contour_lines(&mut self, threshold: f32) -> Vec<Tensor> {
        self.tiles.init_threshold(threshold);

        self.tiles.find_contours()
    }

    #[cfg(test)]
    fn tile(&self, index: usize) -> &Tile {
        &self.tiles.tiles[index]
    }
}

pub struct Tiles {
    xy: Tensor,
    tiles: Vec<Tile>,
    z: Tensor,
}

impl Tiles {
    pub fn new(tri: &Triangulation, z: Tensor) -> Self {
        let mut tri_map = HashMap::<VertId, Vec<TileId>>::new();

        let mut tiles = Vec::<Tile>::new();

        for (i, verts) in tri.triangles().iter_slice().enumerate() {
            let a = VertId(verts[0]);
            let b = VertId(verts[1]);
            let c = VertId(verts[2]);

            let id = TileId(i);
            let tile = Tile::new(id, [a, b, c]);

            add_dual_map(&mut tri_map, &tile);

            tiles.push(tile);
        }

        for i in 0..tiles.len() {
            let [a, b, c] = tiles[i].verts;

            let ab_dual = find_dual(&tiles, &tri_map, b, a);
            let bc_dual = find_dual(&tiles, &tri_map, c, b);
            let ca_dual = find_dual(&tiles, &tri_map, a, c);

            let tile = &mut tiles[i];
            tile.ab_dual = ab_dual;
            tile.bc_dual = bc_dual;
            tile.ca_dual = ca_dual;
        }

        Self {
            xy: tri.vertices().clone(),
            tiles,
            z,
        }
    }

    pub fn init_threshold(&mut self, threshold: f32) {
        for tile in &mut self.tiles {
            tile.init_threshold(threshold, &self.z);
        }
    }

    pub fn find_contours(&mut self) -> Vec<Tensor> {
        let mut contours = Vec::<Tensor>::new();

        for i in 0..self.tiles.len() {
            let id = TileId(i);

            if let Some(contour) = self.find_contour(id) {
                contours.push(contour);
            }
        }

        contours
    }

    ///
    /// Contour if one of the edges has a cross.
    /// 
    fn find_contour(&mut self, id: TileId) -> Option<Tensor> {
        // let tile = &self[id];

        if self[id].ab.is_cross() {
            let ab = self[id].ab;
            let (a, b) = (self[id].verts[0], self[id].verts[1]);
            let dual = self[id].ab_dual;

            let mut path = Cursor::path_ab(self, id, a, b);

            if let Some(dual_path) = Cursor::path_dual(self, dual, b, a) {
                let mut dual_path = dual_path;
                path.reverse();
                path.pop();
                path.append(&mut dual_path);
            };

            if ab == CrossEdge::Up {
                path.reverse();
            };

            Some(path.into_tensor())
        } else if self[id].bc.is_cross() {
            let bc = self[id].bc;
            let (b, c) = (self[id].verts[1], self[id].verts[2]);
            let dual = self[id].bc_dual;
    
            let mut path = Cursor::path_bc(self, id, b, c);
    
            if let Some(dual_path) = Cursor::path_dual(self, dual, c, b) {
                let mut dual_path = dual_path;
                path.reverse();
                path.pop();
                path.append(&mut dual_path);
            };

            if bc == CrossEdge::Up {
                path.reverse();
            };
    
            Some(path.into_tensor())
        } else if self[id].ca.is_cross() {
            let ca = self[id].ca;
            let (c, a) = (self[id].verts[2], self[id].verts[0]);
            let dual = self[id].ca_dual;
    
            let mut path = Cursor::path_ca(self, id, c, a);
    
            if let Some(dual_path) = Cursor::path_dual(self, dual, a, c) {
                let mut dual_path = dual_path;
                path.reverse();
                path.pop();
                path.append(&mut dual_path);
            };

            if ca == CrossEdge::Up {
                path.reverse();
            };
    
            Some(path.into_tensor())
        } else {
            None
        }
    }

    fn clear_from_dual(&mut self, id: TileId, v0: VertId, v1: VertId) -> bool {
        if ! id.is_none() {
            self[id].clear_from_dual(v0, v1);

            true
        } else {
            false
        }
    }

    #[inline]
    fn vertex(&self, id: VertId) -> [f32; 2] {
        [self.xy[(id.i(), 0)], self.xy[(id.i(), 1)]]
    }

}

impl Index<TileId> for Tiles {
    type Output = Tile;

    fn index(&self, id: TileId) -> &Self::Output {
            &self.tiles[id.i()]
    }
}

impl IndexMut<TileId> for Tiles {
    fn index_mut(&mut self, id: TileId) -> &mut Self::Output {
        &mut self.tiles[id.i()]
    }
}

pub struct Cursor {
    vec: TensorVec<[f32; 2]>,
    id: TileId,
}

impl Cursor {
    fn path_ab(tiles: &mut Tiles, id: TileId, a: VertId, b: VertId) -> TensorVec<[f32; 2]> {
        let mut vec = TensorVec::<[f32; 2]>::new();

        let va = tiles.vertex(a);
        let vb = tiles.vertex(b);

        vec.push(tiles[id].cross_ab(va, vb));

        Self::path(tiles, vec, id)
    }

    fn path_bc(tiles: &mut Tiles, id: TileId, b: VertId, c: VertId) -> TensorVec<[f32; 2]> {
        let mut vec = TensorVec::<[f32; 2]>::new();

        let vb = tiles.vertex(b);
        let vc = tiles.vertex(c);

        vec.push(tiles[id].cross_bc(vb, vc));

        Self::path(tiles, vec, id)
    }

    fn path_ca(tiles: &mut Tiles, id: TileId, c: VertId, a: VertId) -> TensorVec<[f32; 2]> {
        let mut vec = TensorVec::<[f32; 2]>::new();

        let vc = tiles.vertex(c);
        let va = tiles.vertex(a);

        vec.push(tiles[id].cross_ca(vc, va));

        Self::path(tiles, vec, id)
    }

    fn path_dual(tiles: &mut Tiles, id: TileId, v0: VertId, v1: VertId) -> Option<TensorVec<[f32; 2]>> {
        if id.is_none() {
            return None;
        }

        let tile = &tiles[id];

        if tile.verts[0] == v0 && tile.verts[1] == v1 {
            if tile.ab.is_cross() {
                Some(Self::path_ab(tiles, id, v0, v1))
            } else {
                None
            }
        } else if tile.verts[1] == v0 && tile.verts[2] == v1 {
            if tile.bc.is_cross() {
                Some(Self::path_bc(tiles, id, v0, v1))
            } else {
                None
            }
        } else if tile.verts[2] == v0 && tile.verts[0] == v1 {
            if tile.ca.is_cross() {
                Some(Self::path_ca(tiles, id, v0, v1))
            } else {
                None
            }
        } else {
            todo!()
        }
    }

    fn path(tiles: &mut Tiles, vec: TensorVec<[f32; 2]>, id: TileId) -> TensorVec<[f32; 2]> {
        let mut cursor = Self {
            vec,
            id,
        };

        while cursor.next(tiles) {
        }

        cursor.vec
    }

    fn next(&mut self, tiles: &mut Tiles) -> bool {
        let id = self.id;

        let [a, b, c] = tiles[id].verts;

        let pa = tiles.vertex(a);
        let pb = tiles.vertex(b);
        let pc = tiles.vertex(c);

        if tiles[id].ab.is_cross() {
            self.vec.push(tiles[id].cross_ab(pa, pb));

            self.id = tiles[id].ab_dual;

            tiles.clear_from_dual(self.id, b, a)
        } else if tiles[id].bc.is_cross() {
            self.vec.push(tiles[id].cross_bc(pb, pc));

            self.id = tiles[id].bc_dual;

            tiles.clear_from_dual(self.id, c, b)
        } else if tiles[id].ca.is_cross() {
            self.vec.push(tiles[id].cross_ca(pc, pa));

            self.id = tiles[id].ca_dual;

            tiles.clear_from_dual(self.id, a, c)
        } else {
            self.id = TileId::none();

            false
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VertId(usize);

impl VertId {
    #[inline]
    pub fn i(&self) -> usize {
        self.0
    }
}

fn find_dual(
    tiles: &Vec<Tile>,
    map: &HashMap<VertId, Vec<TileId>>, 
    a: VertId, 
    b: VertId
) -> TileId {
    if let Some(tile_ids) = map.get(&a) {
        for dual_id in tile_ids {
            let tile = &tiles[dual_id.i()];

            if tile.verts[0] == a && tile.verts[1] == b {
                //tile.ab_dual = *dual_id;
                return tile.id;
            } else if tile.verts[1] == a && tile.verts[2] == b {
                //tile.bc_dual = *dual_id;
                return tile.id;
            } else if tile.verts[2] == a && tile.verts[0] == b {
                //tile.ca_dual = *dual_id;
                return tile.id;
            }
        }
    }

    TileId::none()
}

fn add_dual_map(
    map: &mut HashMap<VertId, Vec<TileId>>,
    tile: &Tile
) {
    add_dual_map_item(map, tile.id, tile.verts[0]);
    add_dual_map_item(map, tile.id, tile.verts[1]);
    add_dual_map_item(map, tile.id, tile.verts[2]);
}

fn add_dual_map_item(
    map: &mut HashMap<VertId, Vec<TileId>>,
    id: TileId,
    vert: VertId,
) {
    if let Some(tiles) = map.get_mut(&vert) {
        tiles.push(id);
    } else {
        map.insert(vert, vec![id]);
    }
}

#[cfg(test)]
mod test {
    use essay_tensor::{tf32, tensor::TensorVec};

    use crate::{contour::tritile::CrossEdge, tri::Triangulation};

    use super::{TriContourGenerator};

    fn gen_new(xyz: Vec<[f32; 3]>, tri: Vec<[u32; 3]>) -> TriContourGenerator {
        let mut vec_xy = TensorVec::<[f32; 2]>::new();
        let mut vec_z = TensorVec::<f32>::new();

        for [x, y, z] in xyz {
            vec_xy.push([x, y]);
            vec_z.push(z);
        }

        let mut vec_tri = TensorVec::<[usize; 3]>::new();

        for tri in tri {
            vec_tri.push([tri[0] as usize, tri[1] as usize, tri[2] as usize]);
        }

        let tri = Triangulation::new(
            vec_xy.into_tensor(), 
            vec_tri.into_tensor()
        );

        TriContourGenerator::new(&tri, vec_z.into_tensor())
    }

    // single title bisected by a cross
    #[test]
    fn single_tile_above_below() {
        let mut cg = gen_new(
            vec![[0., 0., 0.], [2., 0., 0.], [1., 1., 0.]],
            vec![[0, 1, 2]], 
        );

        cg.init_threshold(2.);

        assert_eq!(cg.tile(0).ab, CrossEdge::Below);
        assert_eq!(cg.tile(0).bc, CrossEdge::Below);
        assert_eq!(cg.tile(0).ca, CrossEdge::Below);

        cg.init_threshold(-2.);

        assert_eq!(cg.tile(0).ab, CrossEdge::Above);
        assert_eq!(cg.tile(0).bc, CrossEdge::Above);
        assert_eq!(cg.tile(0).ca, CrossEdge::Above);

        cg.init_threshold(0.);

        assert_eq!(cg.tile(0).ab, CrossEdge::Above);
        assert_eq!(cg.tile(0).bc, CrossEdge::Above);
        assert_eq!(cg.tile(0).ca, CrossEdge::Above);
    }

    // single title bisected by a cross
    #[test]
    fn single_tile_cross() {
        let mut cg = gen_new(
            vec![[0., 0., 1.], [2., 0., 0.], [1., 1., 0.]],
            vec![[0, 1, 2]], 
        );

        cg.init_threshold(0.5);

        assert_eq!(cg.tile(0).ab, CrossEdge::Down);
        assert_eq!(cg.tile(0).bc, CrossEdge::Below);
        assert_eq!(cg.tile(0).ca, CrossEdge::Up);

        let mut cg = gen_new(
            vec![[0., 0., 0.], [2., 0., 1.], [1., 1., 0.]],
            vec![[0, 1, 2]], 
        );

        cg.init_threshold(0.5);

        assert_eq!(cg.tile(0).ab, CrossEdge::Up);
        assert_eq!(cg.tile(0).bc, CrossEdge::Down);
        assert_eq!(cg.tile(0).ca, CrossEdge::Below);

        let mut cg = gen_new(
            vec![[0., 0., 0.], [2., 0., 0.], [1., 1., 1.]],
            vec![[0, 1, 2]], 
        );

        cg.init_threshold(0.5);

        assert_eq!(cg.tile(0).ab, CrossEdge::Below);
        assert_eq!(cg.tile(0).bc, CrossEdge::Up);
        assert_eq!(cg.tile(0).ca, CrossEdge::Down);

        let mut cg = gen_new(
            vec![[0., 0., 0.], [2., 0., 1.], [1., 1., 1.]],
            vec![[0, 1, 2]], 
        );

        cg.init_threshold(0.5);

        assert_eq!(cg.tile(0).ab, CrossEdge::Up);
        assert_eq!(cg.tile(0).bc, CrossEdge::Above);
        assert_eq!(cg.tile(0).ca, CrossEdge::Down);

        let mut cg = gen_new(
            vec![[0., 0., 1.], [2., 0., 0.], [1., 1., 1.]],
            vec![[0, 1, 2]], 
        );

        cg.init_threshold(0.5);

        assert_eq!(cg.tile(0).ab, CrossEdge::Down);
        assert_eq!(cg.tile(0).bc, CrossEdge::Up);
        assert_eq!(cg.tile(0).ca, CrossEdge::Above);

        let mut cg = gen_new(
            vec![[0., 0., 1.], [2., 0., 1.], [1., 1., 0.]],
            vec![[0, 1, 2]], 
        );

        cg.init_threshold(0.5);

        assert_eq!(cg.tile(0).ab, CrossEdge::Above);
        assert_eq!(cg.tile(0).bc, CrossEdge::Down);
        assert_eq!(cg.tile(0).ca, CrossEdge::Up);
    }

    /// no contours when no crossings
    #[test]
    fn contour_zero_tile() {
        let mut cg = gen_new(
            vec![[0., 0., 0.], [2., 0., 0.], [1., 1., 0.]],
            vec![[0, 1, 2]], 
        );

        let contour = cg.contour_lines(0.5);
        assert_eq!(contour.len(), 0);

        let mut cg = gen_new(
            vec![[0., 0., 1.], [2., 0., 1.], [1., 1., 1.]],
            vec![[0, 1, 2]], 
        );

        let contour = cg.contour_lines(0.5);
        assert_eq!(contour.len(), 0);
    }

    /// contours for a single tile
    #[test]
    fn contour_single_tri() {
        let mut cg = gen_new(
            vec![[0., 0., 1.], [2., 0., 0.], [0., 2., 0.]],
            vec![[0, 1, 2]], 
        );
        assert_eq!(
            cg.contour_lines(0.5),
            vec![tf32!([[1., 0.], [0., 1.]])]
        );

        let mut cg = gen_new(
            vec![[0., 0., 0.], [2., 0., 1.], [0., 2., 0.]],
            vec![[0, 1, 2]], 
        );
        assert_eq!(
            cg.contour_lines(0.5),
            vec![tf32!([[1., 1.], [1., 0.]])]
        );

        let mut cg = gen_new(
            vec![[0., 0., 0.], [2., 0., 0.], [0., 2., 1.]],
            vec![[0, 1, 2]], 
        );
        assert_eq!(
            cg.contour_lines(0.5),
            vec![tf32!([[0., 1.], [1., 1.]])]
        );

        let mut cg = gen_new(
            vec![[0., 0., 0.], [2., 0., 1.], [0., 2., 1.]],
            vec![[0, 1, 2]], 
        );
        assert_eq!(
            cg.contour_lines(0.5),
            vec![tf32!([[0., 1.], [1., 0.]])]
        );

        let mut cg = gen_new(
            vec![[0., 0., 1.], [2., 0., 0.], [0., 2., 1.]],
            vec![[0, 1, 2]], 
        );
        assert_eq!(
            cg.contour_lines(0.5),
            vec![tf32!([[1., 0.], [1., 1.]])]
        );

        let mut cg = gen_new(
            vec![[0., 0., 1.], [2., 0., 1.], [0., 2., 0.]],
            vec![[0, 1, 2]], 
        );
        assert_eq!(
            cg.contour_lines(0.5),
            vec![tf32!([[1., 1.], [0., 1.]])]
        );

    }

    /// contours for a quad tile
    #[test]
    fn contour_quad_tile_single_corner() {
        let mut cg = gen_new(
            vec![[0., 0., 1.], [2., 0., 0.], [2., 2., 0.], [0., 2., 0.]],
            vec![[0, 1, 2], [0, 2, 3]], 
        );
        assert_eq!(
            cg.contour_lines(0.5),
            vec![tf32!([[1., 0.], [1., 1.], [0., 1.]])]
        );

        let mut cg = gen_new(
            vec![[0., 0., 0.], [2., 0., 1.], [2., 2., 0.], [0., 2., 0.]],
            vec![[0, 1, 2], [0, 2, 3]], 
        );
        assert_eq!(
            cg.contour_lines(0.5),
            vec![tf32!([[2., 1.], [1., 0.]])]
        );

        let mut cg = gen_new(
            vec![[0., 0., 0.], [2., 0., 0.], [2., 2., 1.], [0., 2., 0.]],
            vec![[0, 1, 2], [0, 2, 3]], 
        );
        assert_eq!(
            cg.contour_lines(0.5),
            vec![tf32!([[1., 2.], [1., 1.], [2., 1.]])]
        );

        let mut cg = gen_new(
            vec![[0., 0., 0.], [2., 0., 0.], [2., 2., 0.], [0., 2., 1.]],
            vec![[0, 1, 2], [0, 2, 3]], 
        );
        assert_eq!(
            cg.contour_lines(0.5),
            vec![tf32!([[0., 1.], [1., 2.]])]
        );

        // flip direction/values

        let mut cg = gen_new(
            vec![[0., 0., 0.], [2., 0., 1.], [2., 2., 1.], [0., 2., 1.]],
            vec![[0, 1, 2], [0, 2, 3]], 
        );
        assert_eq!(
            cg.contour_lines(0.5),
            vec![tf32!([[0., 1.], [1., 1.], [1., 0.]])]
        );

        let mut cg = gen_new(
            vec![[0., 0., 1.], [2., 0., 0.], [2., 2., 1.], [0., 2., 1.]],
            vec![[0, 1, 2], [0, 2, 3]], 
        );
        assert_eq!(
            cg.contour_lines(0.5),
            vec![tf32!([[1., 0.], [2., 1.]])]
        );

        let mut cg = gen_new(
            vec![[0., 0., 1.], [2., 0., 1.], [2., 2., 0.], [0., 2., 1.]],
            vec![[0, 1, 2], [0, 2, 3]], 
        );
        assert_eq!(
            cg.contour_lines(0.5),
            vec![tf32!([[2., 1.], [1., 1.], [1., 2.]])]
        );

        let mut cg = gen_new(
            vec![[0., 0., 1.], [2., 0., 1.], [2., 2., 1.], [0., 2., 0.]],
            vec![[0, 1, 2], [0, 2, 3]], 
        );
        assert_eq!(
            cg.contour_lines(0.5),
            vec![tf32!([[1., 2.], [0., 1.]])]
        );
    }

    /// contours for a quad tile where one side is higher/lower
    #[test]
    fn contour_quad_tile_side() {
        let mut cg = gen_new(
            vec![[0., 0., 1.], [2., 0., 1.], [2., 2., 0.], [0., 2., 0.]],
            vec![[0, 1, 2], [0, 2, 3]], 
        );
        assert_eq!(
            cg.contour_lines(0.5),
            vec![tf32!([[2., 1.], [1., 1.], [0., 1.]])]
        );

        let mut cg = gen_new(
            vec![[0., 0., 1.], [2., 0., 0.], [2., 2., 0.], [0., 2., 1.]],
            vec![[0, 1, 2], [0, 2, 3]], 
        );
        assert_eq!(
            cg.contour_lines(0.5),
            vec![tf32!([[1., 0.], [1., 1.], [1., 2.]])]
        );

        // flip low/high
        let mut cg = gen_new(
            vec![[0., 0., 0.], [2., 0., 0.], [2., 2., 1.], [0., 2., 1.]],
            vec![[0, 1, 2], [0, 2, 3]], 
        );
        assert_eq!(
            cg.contour_lines(0.5),
            vec![tf32!([[0., 1.], [1., 1.], [2., 1.]])]
        );

        let mut cg = gen_new(
            vec![[0., 0., 0.], [2., 0., 1.], [2., 2., 1.], [0., 2., 0.]],
            vec![[0, 1, 2], [0, 2, 3]], 
        );
        assert_eq!(
            cg.contour_lines(0.5),
            vec![tf32!([[1., 2.], [1., 1.], [1., 0.]])]
        );
    }

    /// contours for a quad tile where opposite corners are equal
    #[test]
    fn contour_quad_tile_cross_corner() {
        let mut cg = gen_new(
            vec![[0., 0., 1.], [2., 0., 0.], [2., 2., 1.], [0., 2., 0.]],
            vec![[0, 1, 2], [0, 2, 3]], 
        );
        assert_eq!(
            cg.contour_lines(0.5),
            vec![
                tf32!([[1., 0.], [1., 1.], [0., 1.]]),
                tf32!([[1., 2.], [1., 1.], [2., 1.]]),
                ]
        );

        let mut cg = gen_new(
            vec![[0., 0., 0.], [2., 0., 1.], [2., 2., 0.], [0., 2., 1.]],
            vec![[0, 1, 2], [0, 2, 3]], 
        );
        assert_eq!(
            cg.contour_lines(0.5),
            vec![
                tf32!([[2., 1.], [1., 0.]]),
                tf32!([[0., 1.], [1., 2.]]),
                ]
        );
    }

    /*

    /// contour starting from bottom for (n, 0)
    #[test]
    fn contour_start_bottom() {
        // SE
        let mut cg = ContourGenerator::new(tf32!([
            [1., 1., 0.], [1., 1., 1.]
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[1.5, 0.0], [2.0, 0.5]])
        ]);

        // reverse
        let mut cg = ContourGenerator::new(tf32!([
            [0., 0., 1.], [0., 0., 0.]
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[2.0, 0.5], [1.5, 0.0]])
        ]);

        // SN
        let mut cg = ContourGenerator::new(tf32!([
            [1., 1., 0.], [1., 1., 0.]
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[1.5, 0.0], [1.5, 1.0]])
        ]);
    }

    // contour starting from left for (0, 0)
    #[test]
    fn contour_start_left_bl() {
        // EW
        let mut cg = ContourGenerator::new(tf32!([
            [0., 0.], [1., 1.]
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[0.0, 0.5], [1.0, 0.5]])
        ]);

        // reverse - ccw around higher
        let mut cg = ContourGenerator::new(tf32!([
            [1., 1.], [0., 0.]
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[1.0, 0.5], [0.0, 0.5]])
        ]);

        // EN
        let mut cg = ContourGenerator::new(tf32!([
            [0., 0.], [1., 0.]
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[0.0, 0.5], [0.5, 1.0]])
        ]);

        // reverse
        let mut cg = ContourGenerator::new(tf32!([
            [1., 1.], [0., 1.]
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[0.5, 1.0], [0.0, 0.5]])
        ]);
    }

    // initial left path logic for (0, j)
    #[test]
    fn contour_init_left() {
        // EN
        let mut cg = ContourGenerator::new(tf32!([
            [0., 0.], [0., 0.], [1., 0.]
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[0.0, 1.5], [0.5, 2.0]])
        ]);

        // reverse
        let mut cg = ContourGenerator::new(tf32!([
            [1., 1.], [1., 1.], [0., 1.]
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[0.5, 2.0], [0.0, 1.5]])
        ]);

        // EW
        let mut cg = ContourGenerator::new(tf32!([
            [0., 0.], [0., 0.], [1., 1.]
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[0.0, 1.5], [1.0, 1.5]])
        ]);
    }

    #[test]
    fn contour_standard_bl() {
        // NE
        let mut cg = ContourGenerator::new(tf32!([
            [0., 0.], [0., 1.]
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[0.5, 1.0], [1.0, 0.5]])
        ]);

        // reverse
        let mut cg = ContourGenerator::new(tf32!([
            [1., 1.], [1., 0.]
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[1.0, 0.5], [0.5, 1.0]])
        ]);
    }

    #[test]
    fn next_from_left() {
        // EN
        let mut cg = ContourGenerator::new(tf32!([
            [0., 0.], [1., 0.], [1., 0.]
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[0.0, 0.5], [0.5, 1.0], [0.5, 2.0]])
        ]);

        // EW
        let mut cg = ContourGenerator::new(tf32!([
            [0., 0., 0.], [1., 1., 1.],
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[0.0, 0.5], [1.0, 0.5], [2.0, 0.5]])
        ]);

        // ES
        let mut cg = ContourGenerator::new(tf32!([
            [0., 0., 1.], [1., 1., 1.],
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[0.0, 0.5], [1.0, 0.5], [1.5, 0.0]])
        ]);
    }

    #[test]
    fn next_from_bottom() {
        // SW
        let mut cg = ContourGenerator::new(tf32!([
            [1., 0.], [1., 0.], [0., 0.]
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[0.5, 0.0], [0.5, 1.0], [0.0, 1.5]])
        ]);

        // SN
        let mut cg = ContourGenerator::new(tf32!([
            [1., 0.], [1., 0.],
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[0.5, 0.0], [0.5, 1.0]]),
        ]);

        // SE
        let mut cg = ContourGenerator::new(tf32!([
            [1., 0.], [1., 1.],
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[0.5, 0.0], [1.0, 0.5]]),
        ]);
    }

    #[test]
    fn next_from_top() {
        // NE
        let mut cg = ContourGenerator::new(tf32!([
            [1., 0., 0.], [1., 0., 1.], [1., 1., 1.],
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[0.5, 0.0], [0.5, 1.0], [1.0, 1.5], [1.5, 1.0], [2.0, 0.5]])
        ]);

        // NS
        let mut cg = ContourGenerator::new(tf32!([
            [1., 0., 1.], [1., 0., 1.], [1., 1., 1.],
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[0.5, 0.0], [0.5, 1.0], [1.0, 1.5], [1.5, 1.0], [1.5, 0.0]])
        ]);

        // NW
        let mut cg = ContourGenerator::new(tf32!([
            [1., 1., 0.], [1., 1., 0.], [0., 1., 0.], [0., 0., 0.]
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([
                [1.5, 0.0],
                [1.5, 1.0],
                [1.5, 2.0],
                [1.0, 2.5],
                [0.5, 2.0],
                [0.0, 1.5],
                ])
        ]);
    }

    #[test]
    fn test_loop() {
        let mut cg = ContourGenerator::new(tf32!([
            [0., 0., 0.], [0., 1., 0.], [0., 0., 0.],
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([
                [0.5, 1.0],
                [1.0, 0.5],
                [1.5, 1.0],
                [1.0, 1.5],
                [0.5, 1.0],
                ])
        ]);

        // reverse
        let mut cg = ContourGenerator::new(tf32!([
            [1., 1., 1.], [1., 0., 1.], [1., 1., 1.],
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([
                [0.5, 1.0],
                [1.0, 1.5],
                [1.5, 1.0],
                [1.0, 0.5],
                [0.5, 1.0],
                ])
        ]);
    }

    /// notch in bottom to test bottom-left of second loop
    #[test]
    fn test_loop_notch() {
        let mut cg = ContourGenerator::new(tf32!([
            [0., 0., 0., 0., 0.], 
            [0., 1., 0., 1., 0.], 
            [0., 1., 1., 1., 0.],
            [0., 0., 0., 0., 0.], 
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([
                [0.5, 1.0],
                [1.0, 0.5],
                [1.5, 1.0],
                [2.0, 1.5],
                [2.5, 1.0],
                [3.0, 0.5],
                [3.5, 1.0],
                [3.5, 2.0],
                [3.0, 2.5],
                [2.0, 2.5],
                [1.0, 2.5],
                [0.5, 2.0],
                [0.5, 1.0]
                ])
        ]);
    }

    /// notch in bottom to test bottom-left of second loop
    #[test]
    fn two_paths() {
        let mut cg = ContourGenerator::new(tf32!([
            [1., 0., 1.], 
            [0., 0., 0.], 
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[0.5, 0.0], [0.0, 0.5]]),
            tf32!([[2.0, 0.5], [1.5, 0.0]]),
        ]);

        // reverse
        let mut cg = ContourGenerator::new(tf32!([
            [0., 1., 0.], 
            [1., 1., 1.], 
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[0.0, 0.5], [0.5, 0.0]]),
            tf32!([[1.5, 0.0], [2.0, 0.5]]),
        ]);
    }

    /// top right arc is distinct because the standard cell starts in the
    /// middle of a path and needs to stitch the two pieces together
    #[test]
    fn right_top_arc() {
        let mut cg = ContourGenerator::new(tf32!([
            [0., 0., 0.], 
            [0., 1., 1.], 
            [0., 1., 1.],
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([
                [0.5, 2.0],
                [0.5, 1.0],
                [1.0, 0.5],
                [2.0, 0.5],
            ]),
        ]);

        // reverse
        let mut cg = ContourGenerator::new(tf32!([
            [1., 1., 1.], 
            [1., 0., 0.], 
            [1., 0., 0.],
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([
                [2.0, 0.5],
                [1.0, 0.5],
                [0.5, 1.0],
                [0.5, 2.0],
            ]),
        ]);
    }
    */
}