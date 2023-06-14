use essay_tensor::Tensor;

use crate::{artist::{Lines2d}, graph::{Graph, graph::PlotOpt}};

pub fn plot(
    graph: &mut Graph, 
    x: impl Into<Tensor>, 
    y: impl Into<Tensor>, 
) -> PlotOpt {
    let lines = Lines2d::from_xy(x, y);

    //self.artist(lines)
    graph.add_data_artist(lines)
}

