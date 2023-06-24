use essay_tensor::Tensor;
use crate::{graph::{Graph}, artist::{LinesOpt, ContainerOpt}};

mod pcolormesh;
mod contour;
mod bar;
mod pie;
mod scatter;
mod plot;

pub use bar::{
    bar_y, BarOpt,
};

pub use pcolormesh::{
    pcolormesh, 
};

pub use contour::{
    contour, 
};

pub use plot::{
    plot, 
};

pub use pie::{
    pie, 
};

pub use scatter::{
    scatter, ScatterOpt,
};

impl Graph {
    pub fn plot(
        &mut self, 
        x: impl Into<Tensor>,
        y: impl Into<Tensor>,
    ) -> LinesOpt {
        plot::plot(self, x, y)
    }

    pub fn scatter(
        &mut self, 
        x: impl Into<Tensor>,
        y: impl Into<Tensor>,
    ) -> ScatterOpt {
        scatter::scatter(self, x, y)
    }

    pub fn pie(
        &mut self,
        x: impl Into<Tensor>, 
    ) -> ContainerOpt {
        pie::pie(self, x)
    }
    /*
    pub fn bar_y(
        &mut self, 
        y: impl Into<Tensor>,
    ) -> BarOpt {
        bar::bar_y(self, y)
    }
    */

}
