use essay_tensor::Tensor;

use crate::{graph::Graph, artist::{ColorMesh}};

pub fn pcolormesh(
    graph: &mut Graph, 
    data: impl Into<Tensor>,
) {
    let colormesh = ColorMesh::new(data);
    
    graph.add_simple_artist(colormesh);
}
