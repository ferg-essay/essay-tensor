use essay_tensor::{Tensor, tensor::TensorVec};

use super::triangulate;

pub struct Triangulation {
    xy: Tensor,
    triangles: Tensor<usize>,
}

impl Triangulation {
    pub fn new(xy: Tensor, triangles: Tensor<usize>) -> Self {
        assert!(xy.rank() == 2, "xy must be a 2D list (rank-2) {:?}", xy.shape().as_slice());
        assert!(xy.cols() == 2, "xy must be a 2D list (rank-2) {:?}", xy.shape().as_slice());

        assert!(triangles.rank() == 2, "triangles must be a list of triple indices (rank-2) {:?}", xy.shape().as_slice());
        assert!(triangles.cols() == 3, "triangles must be a list of triple indices (rank-2) {:?}", xy.shape().as_slice());

        Self {
            xy,
            triangles,
        }
    }

    pub fn vertices(&self) -> &Tensor {
        &self.xy
    }

    pub fn triangles(&self) -> &Tensor<usize> {
        &self.triangles
    }

    pub fn edges(&self) -> Tensor<usize> {
        let mut edges = TensorVec::<[usize; 2]>::new();

        for triangle in self.triangles.iter_slice() {
            let (a, b, c) = (triangle[0], triangle[1], triangle[2]);

            if a < b {
                edges.push([a, b]);
            } else {
                edges.push([b, a]);
            }

            if b < c {
                edges.push([b, c]);
            } else {
                edges.push([c, b]);
            }

            if c < a {
                edges.push([c, a]);
            } else {
                edges.push([a, c]);
            }
        }

        edges.into_tensor()
    }
}

impl From<Tensor> for Triangulation {
    fn from(value: Tensor) -> Self {
        triangulate(&value)
    }
}

impl From<&Tensor> for Triangulation {
    fn from(value: &Tensor) -> Self {
        triangulate(value)
    }
}