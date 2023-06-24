use essay_tensor::{Tensor, tensor::TensorVec};

use crate::contour::tile::TileGrid;

use super::tile::{CrossEdge, Tile};

pub struct ContourGenerator {
    z: Tensor,
    tile_cols: usize,
    tile_rows: usize,
    tiles: TileGrid,
}

impl ContourGenerator {
    pub fn new(z: Tensor) -> Self {
        assert!(z.rank() == 2, "contour z data must be 2D (rank 2) {:?}", z.shape().as_slice());
        assert!(z.cols() > 0, "contour z data must have size > 0 {:?}", z.shape().as_slice());
        assert!(z.rows() > 0, "contour z data must have size > 0 {:?}", z.shape().as_slice());

        let tile_cols = z.cols() - 1;
        let tile_rows = z.rows() - 1;

        let tiles = TileGrid::new(tile_cols, tile_rows);

        Self {
            z,
            tile_cols,
            tile_rows,
            tiles,
        }
    }

    pub fn contour_lines(&mut self, threshold: f32) -> Vec<Tensor> {
        self.tiles.clear();
        self.build_edges(threshold);

        self.find_contours()
    }

    pub(crate) fn get_tile(&mut self, x: usize, y: usize) -> &Tile {
        &self.tiles[(x, y)]
    }

    /// For each tile, annotate each edge as crossing the threshold or not.
    fn build_edges(&mut self, threshold: f32) {
        let z = &self.z;

        let (rows, cols) = (z.rows(), z.cols());

        for j in 0..rows {
            for i in 0..cols {
                let z0 = z[(j, i)];

                if j + 1 < rows {
                    let z1 = z[(j + 1, i)];

                    let (vert_edge, mp) = CrossEdge::from_z(threshold, z0, z1);

                    if i > 0 {
                        self.tiles[(i - 1, j)].right(vert_edge, mp);
                    }

                    if i + 1 < cols {
                        self.tiles[(i, j)].left(vert_edge, mp);
                    }
                }

                if i + 1 < cols {
                    let z2 = z[(j, i + 1)];

                    let (horiz_edge, mp) = CrossEdge::from_z(threshold, z0, z2);

                    if j + 1 < rows {
                        self.tiles[(i, j)].bottom(horiz_edge, mp);
                    }

                    if j > 0 {
                        self.tiles[(i, j - 1)].top(horiz_edge, mp);
                    }
                }
            }
        }
    }

    fn find_contours(&mut self) -> Vec<Tensor> {
        let mut paths = Vec::<Tensor>::new();

        for i in 0..self.tile_cols {
            if let Some(path) = self.start_bottom(i) {
                paths.push(path);
            }
        }

        for j in 0..self.tile_rows {
            if let Some(path) = self.start_left(j) {
                paths.push(path);
            }

            for i in 0..self.tile_cols {
                if let Some(path) = self.start_center(i, j) {
                    paths.push(path);
                }
            }
        }

        paths
    }

    /// Contour that starts on the bottom edge: (i, 0).
    fn start_bottom(&mut self, i: usize) -> Option<Tensor> {
        let tile = self.get_tile(i, 0);

        let bottom = tile.get_bottom();
        if bottom.is_cross() {
            let mut path = self.path(i, 0, Dir::B);

            if bottom == CrossEdge::Up { // ccw around higher values
                path.reverse();
            }

            Some(path.into_tensor())
        } else {
            None
        }
    }

    /// Contour that starts on the left edge: (0, j)
    fn start_left(&mut self, j: usize) -> Option<Tensor> {
        let tile = self.get_tile(0, j);

        let left = tile.get_left();
        if left.is_cross() {
            let mut path = self.path(0, j, Dir::L);

            if left == CrossEdge::Down { // ccw around higher values
                path.reverse();
            }

            Some(path.into_tensor())
        } else {
            None
        }
    }

    ///
    /// Contour that starts in the middle: (i, j).
    /// 
    /// Because of the search order, the only remaining possibility should 
    /// be top-right. All other directions should already be discovered
    /// 
    fn start_center(&mut self, i: usize, j: usize) -> Option<Tensor> {
        let top = self.get_tile(i, j).get_top();
        if top.is_cross() {
            let cols = self.tile_cols;
            let mut path = if j + 1 == cols { // right side, so follow from edge
                self.path(i, j, Dir::T)
            } else { // crossing top-right
                let mut path = self.path(i, j, Dir::T);

                if self.get_tile(i, j + 1).get_bottom().is_cross() {
                    // non-loop, so must follow other side and stitch together
                    let mut top_path = self.path(i, j + 1, Dir::B);

                    top_path.reverse();
                    top_path.pop();
                    top_path.append(&mut path);
    
                    top_path
                } else {
                    path
                }
            };

            if top == CrossEdge::Down { // ccw around higher values
                path.reverse();
            }

            Some(path.into_tensor())
    } else {
            None
        }
    }

    fn path(&mut self, i: usize, j: usize, source: Dir) -> TensorVec<[f32; 2]> {
        let cursor = Cursor::path(self, i, j, source);

        cursor.to_vec()
    }
}

pub struct Cursor {
    vec: TensorVec<[f32; 2]>,
    init_i: usize,
    init_j: usize,

    i: usize,
    j: usize,
    source: Dir,
}

impl Cursor {
    fn path(gc: &mut ContourGenerator, i: usize, j: usize, source: Dir) -> Self {
        let mut vec = TensorVec::<[f32; 2]>::new();

        match source {
            Dir::T => {
                let t = gc.tiles[(i, j)].top_t;
                vec.push([i as f32 + t, (j + 1) as f32])
            }
            Dir::B => {
                let t = gc.tiles[(i, j)].bottom_t;
                vec.push([i as f32 + t, j as f32])
            },
            Dir::R => {
                let t = gc.tiles[(i, j)].right_t;
                vec.push([(i + 1) as f32, j as f32 + t])
            },
            Dir::L => {
                let t = gc.tiles[(i, j)].left_t;
                vec.push([i as f32, j as f32 + t])
            },
        }

        let mut cursor = Self {
            vec,
            init_i: i,
            init_j: j,
            i,
            j,
            source,
        };

        while cursor.next(gc) {
        }

        cursor
    }

    fn next(&mut self, cg: &mut ContourGenerator) -> bool {
        let (i, j) = (self.i, self.j);

        if i == self.init_i && j == self.init_j && self.vec.len() > 1 {
            return false;
        }

        let tile = &mut cg.tiles[(i, j)];

        match self.source {
            Dir::T => {
                tile.cross_top().unwrap();

                if let Some(t) = tile.cross_right() {
                    self.vec.push([(i + 1) as f32, j as f32 + t]);

                    i + 1 < cg.tile_cols && self.set_next(i + 1, j, Dir::L)
                } else if let Some(t) = tile.cross_bottom() {
                    self.vec.push([i as f32 + t, j as f32]);

                    j > 0 && self.set_next(i, j - 1, Dir::T)
                } else if let Some(t) = tile.cross_left() {
                    self.vec.push([i as f32, j as f32 + t]);
    
                    i > 0 && self.set_next(i - 1, j, Dir::R)
                } else {
                    false
                }
            }
            Dir::B => {
                tile.cross_bottom().unwrap();

                if let Some(t) = tile.cross_left() {
                    self.vec.push([i as f32, j as f32 + t]);

                    i > 0 && self.set_next(i - 1, j, Dir::R)
                } else if let Some(t) = tile.cross_top() {
                    self.vec.push([i as f32 + t, (j + 1) as f32]);

                    j + 1 < cg.tile_rows && self.set_next(i, j + 1, Dir::B)
                } else if let Some(t) = tile.cross_right() {
                    self.vec.push([(i + 1) as f32, j as f32 + t]);

                    i + 1 < cg.tile_cols && self.set_next(i + 1, j, Dir::L)
                } else {
                    false
                }
            }
            Dir::R => {
                tile.cross_right().unwrap();

                if let Some(t) = tile.cross_bottom() {
                    self.vec.push([i as f32 + t, j as f32]);

                    j > 0 && self.set_next(i, j - 1, Dir::T)
                } else if let Some(t) = tile.cross_left() {
                    self.vec.push([i as f32, j as f32 + t]);

                    i > 0 && self.set_next(i - 1, j, Dir::R)
                } else if let Some(t) = tile.cross_top() {
                    self.vec.push([i as f32 + t, (j + 1) as f32]);

                    j + 1 < cg.tile_rows && self.set_next(i, j + 1, Dir::B)
                } else {
                    false
                }
            }
            Dir::L => {
                tile.cross_left().unwrap();

                if let Some(t) = tile.cross_top() {
                    self.vec.push([i as f32 + t, (j + 1) as f32]);

                    j + 1 < cg.tile_rows && self.set_next(i, j + 1, Dir::B)
                } else if let Some(t) = tile.cross_right() {
                    self.vec.push([(i + 1) as f32, j as f32 + t]);

                    i + 1 < cg.tile_cols && self.set_next(i + 1, j, Dir::L)
                } else if let Some(t) = tile.cross_bottom() {
                    self.vec.push([i as f32 + t, j as f32]);

                    j > 0 && self.set_next(i, j - 1, Dir::T)
                } else {
                    false
                }
            }
        }
    }

    #[inline]
    fn set_next(&mut self, i: usize, j: usize, dir: Dir) -> bool {
        self.i = i;
        self.j = j;
        self.source = dir;
        true
    }

    fn to_path(self) -> Tensor {
        self.vec.into_tensor()
    }

    fn to_vec(self) -> TensorVec<[f32; 2]> {
        self.vec
    }
}

#[derive(Copy, Clone, Debug)]
pub enum Dir {
    T,
    B,
    R,
    L
}

#[cfg(test)]
mod test {
    use essay_tensor::tf32;

    use crate::contour::tile::CrossEdge;

    use super::ContourGenerator;

    // single title bisected by a cross
    #[test]
    fn single_tile_half() {
        let mut cg = ContourGenerator::new(tf32!([
            [0., 0.], [1., 1.]
        ]));

        cg.build_edges(0.5);

        assert_eq!(cg.get_tile(0, 0).get_left(), CrossEdge::Up);
        assert_eq!(cg.get_tile(0, 0).get_top(), CrossEdge::Above);
        assert_eq!(cg.get_tile(0, 0).get_bottom(), CrossEdge::Below);
        assert_eq!(cg.get_tile(0, 0).get_right(), CrossEdge::Up);

        let mut cg = ContourGenerator::new(tf32!([
            [1., 1.], [0., 0.]
        ]));

        cg.build_edges(0.5);

        assert_eq!(cg.get_tile(0, 0).get_left(), CrossEdge::Down);
        assert_eq!(cg.get_tile(0, 0).get_top(), CrossEdge::Below);
        assert_eq!(cg.get_tile(0, 0).get_bottom(), CrossEdge::Above);
        assert_eq!(cg.get_tile(0, 0).get_right(), CrossEdge::Down);

        let mut cg = ContourGenerator::new(tf32!([
            [0., 1.], [0., 1.]
        ]));

        cg.build_edges(0.5);

        assert_eq!(cg.get_tile(0, 0).get_left(), CrossEdge::Below);
        assert_eq!(cg.get_tile(0, 0).get_top(), CrossEdge::Up);
        assert_eq!(cg.get_tile(0, 0).get_bottom(), CrossEdge::Up);
        assert_eq!(cg.get_tile(0, 0).get_right(), CrossEdge::Above);

        let mut cg = ContourGenerator::new(tf32!([
            [1., 0.], [1., 0.]
        ]));

        cg.build_edges(0.5);

        assert_eq!(cg.get_tile(0, 0).get_left(), CrossEdge::Above);
        assert_eq!(cg.get_tile(0, 0).get_top(), CrossEdge::Down);
        assert_eq!(cg.get_tile(0, 0).get_bottom(), CrossEdge::Down);
        assert_eq!(cg.get_tile(0, 0).get_right(), CrossEdge::Below);
    }

    #[test]
    fn single_tile_corner() {
        let mut cg = ContourGenerator::new(tf32!([
            [0., 1.], [1., 1.]
        ]));

        cg.build_edges(0.5);

        assert_eq!(cg.get_tile(0, 0).get_left(), CrossEdge::Up);
        assert_eq!(cg.get_tile(0, 0).get_top(), CrossEdge::Above);
        assert_eq!(cg.get_tile(0, 0).get_bottom(), CrossEdge::Up);
        assert_eq!(cg.get_tile(0, 0).get_right(), CrossEdge::Above);

        let mut cg = ContourGenerator::new(tf32!([
            [1., 0.], [0., 0.]
        ]));

        cg.build_edges(0.5);

        assert_eq!(cg.get_tile(0, 0).get_left(), CrossEdge::Down);
        assert_eq!(cg.get_tile(0, 0).get_top(), CrossEdge::Below);
        assert_eq!(cg.get_tile(0, 0).get_bottom(), CrossEdge::Down);
        assert_eq!(cg.get_tile(0, 0).get_right(), CrossEdge::Below);

        let mut cg = ContourGenerator::new(tf32!([
            [1., 0.], [1., 1.]
        ]));

        cg.build_edges(0.5);

        assert_eq!(cg.get_tile(0, 0).get_left(), CrossEdge::Above);
        assert_eq!(cg.get_tile(0, 0).get_top(), CrossEdge::Above);
        assert_eq!(cg.get_tile(0, 0).get_bottom(), CrossEdge::Down);
        assert_eq!(cg.get_tile(0, 0).get_right(), CrossEdge::Up);

        let mut cg = ContourGenerator::new(tf32!([
            [0., 1.], [0., 0.]
        ]));

        cg.build_edges(0.5);

        assert_eq!(cg.get_tile(0, 0).get_left(), CrossEdge::Below);
        assert_eq!(cg.get_tile(0, 0).get_top(), CrossEdge::Below);
        assert_eq!(cg.get_tile(0, 0).get_bottom(), CrossEdge::Up);
        assert_eq!(cg.get_tile(0, 0).get_right(), CrossEdge::Down);

        let mut cg = ContourGenerator::new(tf32!([
            [1., 1.], [0., 1.]
        ]));

        cg.build_edges(0.5);

        assert_eq!(cg.get_tile(0, 0).get_left(), CrossEdge::Down);
        assert_eq!(cg.get_tile(0, 0).get_top(), CrossEdge::Up);
        assert_eq!(cg.get_tile(0, 0).get_bottom(), CrossEdge::Above);
        assert_eq!(cg.get_tile(0, 0).get_right(), CrossEdge::Above);

        let mut cg = ContourGenerator::new(tf32!([
            [0., 0.], [1., 0.]
        ]));

        cg.build_edges(0.5);

        assert_eq!(cg.get_tile(0, 0).get_left(), CrossEdge::Up);
        assert_eq!(cg.get_tile(0, 0).get_top(), CrossEdge::Down);
        assert_eq!(cg.get_tile(0, 0).get_bottom(), CrossEdge::Below);
        assert_eq!(cg.get_tile(0, 0).get_right(), CrossEdge::Below);

        let mut cg = ContourGenerator::new(tf32!([
            [1., 1.], [1., 0.]
        ]));

        cg.build_edges(0.5);

        assert_eq!(cg.get_tile(0, 0).get_left(), CrossEdge::Above);
        assert_eq!(cg.get_tile(0, 0).get_top(), CrossEdge::Down);
        assert_eq!(cg.get_tile(0, 0).get_bottom(), CrossEdge::Above);
        assert_eq!(cg.get_tile(0, 0).get_right(), CrossEdge::Down);

        let mut cg = ContourGenerator::new(tf32!([
            [0., 0.], [0., 1.]
        ]));

        cg.build_edges(0.5);

        assert_eq!(cg.get_tile(0, 0).get_left(), CrossEdge::Below);
        assert_eq!(cg.get_tile(0, 0).get_top(), CrossEdge::Up);
        assert_eq!(cg.get_tile(0, 0).get_bottom(), CrossEdge::Below);
        assert_eq!(cg.get_tile(0, 0).get_right(), CrossEdge::Up);
    }

    /// contour starting from bottom for the (0, 0) corner
    #[test]
    fn contour_start_bottom_bl() {
        // SW
        let mut cg = ContourGenerator::new(tf32!([
            [1., 0.], [0., 0.]
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[0.5, 0.0], [0.0, 0.5]])
        ]);

        // reversed direction - ccw around higher
        let mut cg = ContourGenerator::new(tf32!([
            [0., 1.], [1., 1.]
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[0.0, 0.5], [0.5, 0.0]])
        ]);

        // SN
        let mut cg = ContourGenerator::new(tf32!([
            [1., 0.], [1., 0.]
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[0.5, 0.0], [0.5, 1.0]])
        ]);

        // reversed direction
        let mut cg = ContourGenerator::new(tf32!([
            [0., 1.], [0., 1.]
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[0.5, 1.0], [0.5, 0.0]])
        ]);

        // SE
        let mut cg = ContourGenerator::new(tf32!([
            [1., 0.], [1., 1.]
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[0.5, 0.0], [1.0, 0.5]])
        ]);

        // reverse
        let mut cg = ContourGenerator::new(tf32!([
            [0., 1.], [0., 0.]
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[1.0, 0.5], [0.5, 0.0]])
        ]);

        // None
        let mut cg = ContourGenerator::new(tf32!([
            [1., 1.], [1., 1.]
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![]);
    }

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
}