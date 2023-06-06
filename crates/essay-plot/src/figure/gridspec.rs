use crate::figure::CoordMarker;

#[derive(Clone, Debug)]
pub struct GridSpec {
    n_rows: usize,
    n_cols: usize,
}

impl GridSpec {
    pub fn new(n_rows: usize, n_cols: usize) -> Self {
        assert!(n_rows > 0);
        assert!(n_cols > 0);

        Self {
            n_rows,
            n_cols,
        }
    }

    #[inline]
    pub fn rows(&self) -> usize {
        self.n_rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.n_cols
    }
}

impl From<(usize, usize)> for GridSpec {
    fn from(value: (usize, usize)) -> Self {
        GridSpec::new(value.0, value.1)
    }
}

impl From<[usize; 2]> for GridSpec {
    fn from(value: [usize; 2]) -> Self {
        GridSpec::new(value[0], value[1])
    }
}

impl CoordMarker for GridSpec {}


