mod bar;
use essay_tensor::Tensor;

use crate::{graph::{Graph, PlotOpt}};

mod pie;
mod scatter;
mod lineplot;

pub use bar::{
    bar_y, 
};

pub use lineplot::{
    plot, 
};

pub use pie::{
    pie, 
};

pub use scatter::{
    scatter, 
};

impl Graph {
    pub fn plot(
        &mut self, 
        x: impl Into<Tensor>,
        y: impl Into<Tensor>,
    ) -> PlotOpt {
        lineplot::plot(self, x, y)
    }

    pub fn scatter(
        &mut self, 
        x: impl Into<Tensor>,
        y: impl Into<Tensor>,
    ) -> PlotOpt {
        scatter::scatter(self, x, y)
    }

    pub fn pie(
        graph: &mut Graph, 
        x: impl Into<Tensor>, 
    ) -> PlotOpt {
        pie::pie(graph, x)
    }

    pub fn bar_y(
        &mut self, 
        y: impl Into<Tensor>,
    ) -> PlotOpt {
        bar::bar_y(self, y)
    }

}
