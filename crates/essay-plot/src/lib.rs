pub mod artist;
pub mod graph;
pub mod plot;
pub mod driver;

pub mod prelude {
    pub use crate::graph::{Figure};
    pub use crate::plot::{Plot, PlotOpt};
}