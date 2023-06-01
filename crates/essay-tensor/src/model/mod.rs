mod trainer;
mod tape;
mod function;
mod gradient;
mod fit;
mod model;
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

pub use model::{
    ModelBuilder, LayerIn, LayersIn, CallMode,
};

pub(crate) use tape::{
    Tape
};

pub use program::{
    Graph, NodeOp, EvalOp, Operation, IntoForward, BoxForwardOp,
    // BackOp, BoxBackOp,
    TensorCache,
};

pub mod prelude {
    pub use super::fit::{
        Fit, FitOpt
    };
}