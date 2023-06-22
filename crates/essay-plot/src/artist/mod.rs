mod colormap;
mod cycle;
mod pcolor;
mod color;
mod style;
pub mod paths;
mod text;
mod container;
mod markers;
pub mod patch;
mod artist;
mod collection;
mod lines;

pub use artist::{
    Artist, PlotArtist, PlotId,
};

pub use collection::{
    PathCollection
};

pub use container::{
    Container, ContainerOpt
};

pub use color::{
    ColorCycle,
};

pub use colormap::{
    ColorMap,
};

pub use cycle::{
    StyleCycle,
};

pub use pcolor::{
    PColor,
};

pub use lines::{
    Lines2d, LinesOpt, DrawStyle,
};

pub use markers::{
    Markers,
};

pub use patch::{
    PatchTrait,
};

pub use style::{
    PathStyle, 
};

pub use text::{
    Text, // TextStyle,
};
