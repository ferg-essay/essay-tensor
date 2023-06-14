use essay_tensor::Tensor;

use crate::{frame::{Graph, Data}, artist::{Artist, Lines2d}};

pub fn plot(
    graph: &mut Graph, 
    x: impl Into<Tensor>, 
    y: impl Into<Tensor>, 
) -> &mut Artist<Data> {
    let lines = Lines2d::from_xy(x, y);

    //self.artist(lines)
    graph.add_data_artist(lines)
}

