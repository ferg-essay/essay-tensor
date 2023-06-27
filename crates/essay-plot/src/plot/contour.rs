use essay_tensor::Tensor;

use crate::{graph::Graph, artist::{Contour}};

pub fn contour(
    graph: &mut Graph, 
    data: impl Into<Tensor>,
) {
    let contour = Contour::new(data);
    
    graph.add_simple_artist(contour);
}
