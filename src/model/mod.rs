mod trainer;
//mod tape;
mod function;
mod gradient;
// mod fit;
mod tensors;
mod model;
mod var;
pub(crate) mod expr;

pub use var::Var;

pub use function::{
    Function, Tape, ModelId,
};

pub use trainer::Trainer;

pub use model::{
    Model, CallMode, ModelContext,
};
/*
pub(crate) use tape::{
    Tape
};
*/
pub use tensors::Tensors;

pub use expr::{
    Expr, NodeOp, EvalOp, Operation, IntoForward, BoxForwardOp,
    // BackOp, BoxBackOp,
    TensorCache,
};

pub mod prelude {
    //pub use super::fit::{
    //    Fit, FitOpt
    //};
}