mod style;
mod config;
mod figure;
pub mod graph;

pub use graph::{
    Graph,
};

pub use figure::{
    Figure, FigureInner, GraphId,
};

pub use config::{
    Config, ConfigArc,
};

pub use style::{
    PlotOpt,
    //PlotId, // PlotOpt, PlotRef, 
    //PlotArtist, //PathStyleArtist,
};
