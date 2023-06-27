mod colormesh;
mod colorbar;
mod colormaps;
mod colormap;
mod contour;
mod cycle;
mod pcolor;
mod color;
mod style;
mod triplot;
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

pub use colorbar::{
    Colorbar,
};

pub use colormap::{
    ColorMap,
};

pub use colormaps::{
    ColorMaps,
};

pub use colormesh::{
    ColorMesh,
};

pub use contour::{
    Contour,
};

pub use cycle::{
    StyleCycle,
};

pub use pcolor::{
    PColor,
};

pub use triplot::{
    TriPlot,
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
