mod trainer;
mod tape;
mod function;
mod gradient;
mod fit;
mod tensors;
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
    model_builder, ModelBuilder, ModelIn, ModelsIn, CallMode,
};

pub(crate) use tape::{
    Tape
};

pub use tensors::{
    Tensors,
};

pub use program::{
    Program, NodeOp, EvalOp, Operation, IntoForward, BoxForwardOp,
    // BackOp, BoxBackOp,
    TensorCache,
};

pub mod prelude {
    pub use super::fit::{
        Fit, FitOpt
    };
}