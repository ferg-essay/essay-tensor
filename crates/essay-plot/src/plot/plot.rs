use essay_tensor::Tensor;

use crate::{artist::{Lines2d, LinesOpt}, graph::{Graph, PlotOpt}};

pub fn plot(
    graph: &mut Graph, 
    x: impl Into<Tensor>, 
    y: impl Into<Tensor>, 
) -> LinesOpt {
    let lines = Lines2d::from_xy(x, y);

    //self.artist(lines)
    graph.add_plot_artist(lines)
}

