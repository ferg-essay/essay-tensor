mod trainer;
mod tape;
mod function;
mod backprop;
mod var;
pub(crate) mod graph;

pub use var::{
    Var,
};

pub use function::{
    Function,
};

pub use trainer::{
    Trainer,
};

pub(crate) use tape::{
    Tape
};

pub use graph::{
    Graph, NodeOp, EvalOp, Operation, IntoForward, BoxForwardOp,
    // BackOp, BoxBackOp,
    TensorCache,
};
