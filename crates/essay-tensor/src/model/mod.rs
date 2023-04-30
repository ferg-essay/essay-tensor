mod tape;
mod model;
mod var;
mod graph;

pub use var::{
    Var,
};

pub use tape::{
    Tape, TensorId,
};

pub use graph::{
    NodeOp, 
};
