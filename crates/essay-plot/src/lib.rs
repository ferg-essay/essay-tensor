pub mod artist;
pub mod graph;
pub mod figure;
pub mod plot;
pub mod driver;

pub mod prelude {
    //pub use crate::plotly::{Plot, PlotOpt};
    //pub use crate::criterion::{Plot, PlotOpt};
    //pub use crate::egui::{Plot, PlotOpt};
    pub use crate::figure::{Figure};
    pub use crate::plot::{Plot, PlotOpt};
}