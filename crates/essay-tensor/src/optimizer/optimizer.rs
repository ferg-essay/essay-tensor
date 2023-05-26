use crate::{function::Var, Tensor, prelude::Shape};

pub trait Optimizer {
    fn minimize(&self, loss: &Tensor<f32>, vars: &Var<String>, tape: Option<String>);

    fn compute_gradients(&self, loss: &Tensor<f32>, vars: &Vec<String>, tape: Option<String>);

    // Tensor<f32> here is the gradient for the variable
    fn apply_gradients(&self, grads_and_vars: &[(Tensor<f32>, String)]);
}

pub trait OptimizerBuilder {
    fn add_variable(shape: impl Into<Shape>, name: Option<String>) -> Var;
    fn add_variable_from_reference(name: String, var_model: &Var) -> Var;
}