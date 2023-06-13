mod pcolor;
mod color;
mod style;
pub mod paths;
mod text;
mod container;
pub mod patch;
mod artist;
mod collection;
mod lines;

pub use artist::{
    Artist, ArtistTrait
};

pub use collection::{
    Collection
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

//pub use path::{
//    Path, Angle, PathCode,
//};

pub use patch::{
    PatchTrait,
};

//pub use style::{
//    StyleOpt, Style, JoinStyle,
//};

pub use text::{
    Text, // TextStyle,
};
