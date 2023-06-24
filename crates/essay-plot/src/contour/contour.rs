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

        self.build_paths()
    }

    pub(crate) fn get_tile(&mut self, x: usize, y: usize) -> &Tile {
        &self.tiles[(y, x)]
    }

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
                        self.tiles[(j, i - 1)].right(vert_edge, mp);
                    }

                    if i + 1 < cols {
                        self.tiles[(j, i)].left(vert_edge, mp);
                    }
                }

                if i + 1 < cols {
                    let z2 = z[(j, i + 1)];

                    let (horiz_edge, mp) = CrossEdge::from_z(threshold, z0, z2);

                    if j + 1 < rows {
                        self.tiles[(j, i)].bottom(horiz_edge, mp);
                    }

                    if j > 0 {
                        self.tiles[(j - 1, i)].top(horiz_edge, mp);
                    }
                }
            }
        }
    }

    fn build_paths(&mut self) -> Vec<Tensor> {
        let mut paths = Vec::<Tensor>::new();

        for i in 0..self.tile_cols {
            if let Some(path) = self.initial_bottom(i) {
                paths.push(path);
            }
        }

        for j in 0..self.tile_rows {
            if let Some(path) = self.initial_left(j) {
                paths.push(path);
            }

            for i in 0..self.tile_cols {
                if let Some(path) = self.path_start(i, j) {
                    paths.push(path);
                }
            }
        }

        paths
    }

    fn initial_bottom(&mut self, i: usize) -> Option<Tensor> {
        let tile = self.get_tile(i, 0);

        if tile.get_bottom().is_cross() {
            let mut cursor = PathCursor::new(self, i, 0, Dir::B);

            while cursor.next(self) {
            }

            Some(cursor.to_path())
        } else {
            None
        }
    }

    fn initial_left(&mut self, j: usize) -> Option<Tensor> {
        let tile = self.get_tile(0, j);

        if tile.get_left().is_cross() {
            let mut cursor = PathCursor::new(self, 0, j, Dir::L);

            while cursor.next(self) {
            }

            Some(cursor.to_path())
        } else {
            None
        }
    }

    ///
    /// Start detection for each cell. The only possibility should be top-right
    /// because of the search order. All other directions should already be
    /// discovered
    /// 
    fn path_start(&mut self, i: usize, j: usize) -> Option<Tensor> {
        if self.get_tile(i, j).get_top().is_cross() {
            let cols = self.tile_cols;
            if j + 1 == cols {
                let mut cursor = PathCursor::new(self, i, j, Dir::T);

                while cursor.next(self) {
                }
    
                Some(cursor.to_path())
            } else {
                let mut cursor = PathCursor::new(self, i, j, Dir::T);

                while cursor.next(self) {
                }

                if self.get_tile(i, j + 1).get_bottom().is_cross() {
                    let mut right_path = cursor.to_vec();

                    let mut cursor = PathCursor::new(self, j, j + 1, Dir::B);

                    while cursor.next(self) {
                    }

                    let mut top_path = cursor.to_vec();

                    top_path.reverse();
                    top_path.pop();
                    top_path.append(&mut right_path);
    
                    Some(top_path.into_tensor())
                } else { // loop
                    Some(cursor.to_path())
                }
            }
        } else {
            None
        }
    }
}

pub struct PathCursor {
    vec: TensorVec<[f32; 2]>,
    init_i: usize,
    init_j: usize,

    i: usize,
    j: usize,
    source: Dir,
}

impl PathCursor {
    fn new(gc: &ContourGenerator, i: usize, j: usize, source: Dir) -> Self {
        let mut vec = TensorVec::<[f32; 2]>::new();

        match source {
            Dir::T => {
                let t = 0.5;
                vec.push([i as f32 + t, (j + 1) as f32])
            }
            Dir::B => {
                let t = 0.5;
                vec.push([i as f32 + t, j as f32])
            },
            Dir::R => {
                let t = 0.5;
                vec.push([(i + 1) as f32, j as f32 + t])
            },
            Dir::L => {
                let t = 0.5;
                vec.push([i as f32, j as f32 + t])
            },
        }

        Self {
            vec,
            init_i: i,
            init_j: j,
            i,
            j,
            source,
        }
    }

    fn next(&mut self, cg: &mut ContourGenerator) -> bool {
        let (i, j) = (self.i, self.j);

        if i == self.init_i && j == self.init_j && self.vec.len() > 1 {
            return false;
        }

        let tile = &mut cg.tiles[(j, i)];

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

    /// bottom initial logic for the (0, 0) corner
    #[test]
    fn contour_init_bottom_bl() {
        // SW
        let mut cg = ContourGenerator::new(tf32!([
            [1., 0.], [0., 0.]
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[0.5, 0.0], [0.0, 0.5]])
        ]);

        // SN
        let mut cg = ContourGenerator::new(tf32!([
            [0., 1.], [0., 1.]
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[0.5, 0.0], [0.5, 1.0]])
        ]);

        // SE
        let mut cg = ContourGenerator::new(tf32!([
            [0., 1.], [0., 0.]
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[0.5, 0.0], [1.0, 0.5]])
        ]);

        // None
        let mut cg = ContourGenerator::new(tf32!([
            [1., 1.], [1., 1.]
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![]);
    }

    /// bottom initial logic for (n, 0)
    #[test]
    fn contour_init_bottom() {
        // SE
        let mut cg = ContourGenerator::new(tf32!([
            [1., 1., 0.], [1., 1., 1.]
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[1.5, 0.0], [2.0, 0.5]])
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

    // initial left path logic for (0, 0)
    #[test]
    fn contour_init_left_bl() {
        // EW
        let mut cg = ContourGenerator::new(tf32!([
            [0., 0.], [1., 1.]
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[0.0, 0.5], [1.0, 0.5]])
        ]);

        // EN
        let mut cg = ContourGenerator::new(tf32!([
            [0., 0.], [1., 0.]
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[0.0, 0.5], [0.5, 1.0]])
        ]);
    }

    // initial left path logic for (0, j)
    #[test]
    fn contour_init_left() {
        // EN
        let mut cg = ContourGenerator::new(tf32!([
            [1., 1.], [1., 1.], [0., 1.]
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[0.0, 1.5], [0.5, 2.0]])
        ]);

        // EW
        let mut cg = ContourGenerator::new(tf32!([
            [1., 1.], [1., 1.], [0., 0.]
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
            [0., 1.], [0., 0.]
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
            [1., 1.], [0., 1.], [0., 1.]
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[0.0, 0.5], [0.5, 1.0], [0.5, 2.0]])
        ]);

        // EW
        let mut cg = ContourGenerator::new(tf32!([
            [1., 1., 1.], [0., 0., 0.],
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[0.0, 0.5], [1.0, 0.5], [2.0, 0.5]])
        ]);

        // ES
        let mut cg = ContourGenerator::new(tf32!([
            [0., 1., 0.], [0., 0., 0.],
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[0.5, 0.0], [1.0, 0.5], [1.5, 0.0]])
        ]);
    }

    #[test]
    fn next_from_bottom() {
        // SW
        let mut cg = ContourGenerator::new(tf32!([
            [0., 1.], [0., 1.], [1., 1.]
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[0.5, 0.0], [0.5, 1.0], [0.0, 1.5]])
        ]);

        // SN
        let mut cg = ContourGenerator::new(tf32!([
            [0., 1.], [0., 1.],
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[0.5, 0.0], [0.5, 1.0]]),
        ]);

        // SE
        let mut cg = ContourGenerator::new(tf32!([
            [0., 1.], [0., 0.],
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
            [0., 1., 1.], [0., 1., 0.], [0., 0., 0.],
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[0.5, 0.0], [0.5, 1.0], [1.0, 1.5], [1.5, 1.0], [2.0, 0.5]])
        ]);

        // NS
        let mut cg = ContourGenerator::new(tf32!([
            [0., 1., 0.], [0., 1., 0.], [0., 0., 0.],
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
            [1., 1., 1.], [1., 0., 1.], [1., 1., 1.],
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
    }

    /// notch in bottom to test bottom-left of second loop
    #[test]
    fn test_loop_notch() {
        let mut cg = ContourGenerator::new(tf32!([
            [1., 1., 1., 1., 1.], 
            [1., 0., 1., 0., 1.], 
            [1., 0., 0., 0., 1.],
            [1., 1., 1., 1., 1.], 
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([
                [[0.5, 1.0],
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
                [0.5, 1.0]]
                ])
        ]);
    }

    /// notch in bottom to test bottom-left of second loop
    #[test]
    fn two_paths() {
        let mut cg = ContourGenerator::new(tf32!([
            [0., 1., 0.], 
            [1., 1., 1.], 
        ]));

        let lines = cg.contour_lines(0.5);
        assert_eq!(lines, vec![
            tf32!([[0.5, 0.0], [0.0, 0.5]]),
            tf32!([[1.5, 0.0], [2.0, 0.5]]),
        ]);
    }

    /// top right arc is distinct because the standard cell starts in the
    /// middle of a path and needs to stitch the two pieces together
    #[test]
    fn right_top_arc() {
        let mut cg = ContourGenerator::new(tf32!([
            [1., 1., 1.], 
            [1., 0., 0.], 
            [1., 0., 0.],
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
    }
}