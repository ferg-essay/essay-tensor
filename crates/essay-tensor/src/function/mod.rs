mod trainer;
mod tape;
mod function;
mod gradient;
mod var;
pub(crate) mod program;

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

pub use program::{
    Graph, NodeOp, EvalOp, Operation, IntoForward, BoxForwardOp,
    // BackOp, BoxBackOp,
    TensorCache,
};
