use essay_tensor::Tensor;

use crate::{graph::Graph, artist::{ColorMesh, TriPlot, TriContour}, tri::Triangulation};

pub fn tricontour(
    graph: &mut Graph, 
    tri: impl Into<Triangulation>,
    data: impl Into<Tensor>,
) {
    let tricontour = TriContour::new(tri, data);
    
    graph.add_simple_artist(tricontour);
}
