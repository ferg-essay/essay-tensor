use essay_tensor::Tensor;

use crate::{graph::Graph, artist::{ColorMesh, TriPlot}};

pub fn triplot(
    graph: &mut Graph, 
    data: impl Into<Tensor>,
) {
    let triplot = TriPlot::new(data);
    
    graph.add_simple_artist(triplot);
}
