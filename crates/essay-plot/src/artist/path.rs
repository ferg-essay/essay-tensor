use essay_tensor::Tensor;

pub struct Path {
    points: Tensor,
    codes: Vec<PathCode>,
}

impl Path {
    pub fn new(points: impl Into<Tensor>, codes: Vec<PathCode>) -> Path {
        let points = points.into();

        assert_eq!(points.rank(), 2);
        assert_eq!(points.cols(), 2);
        assert_eq!(points.rows(), codes.len());

        // TODO: validate bezier

        Path {
            points,
            codes,
        }
    }
}

pub enum PathCode {
    MoveTo,
    LineTo,
    BezierQuadratic,
    BezierCubic
}