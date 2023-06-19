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
    ArtistStyle, Artist
};

pub use collection::{
    PathCollection
};

pub use container::{
    Container
};

pub use color::{
    ColorCycle,
};

pub use pcolor::{
    PColor,
};

pub use lines::{
    Lines2d,
    Bezier2, Bezier3,
};

pub use markers::{
    Markers,
};

pub use patch::{
    PatchTrait,
};

pub use style::{
    PathStyle, StyleCycle,// StyleChain,
};

pub use text::{
    Text, // TextStyle,
};
