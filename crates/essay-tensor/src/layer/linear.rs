use essay_opt::derive_opt;

use crate::{function::Var, Tensor};

use super::Layer;

pub struct Linear {
    var_a : Var,
    var_b : Option<Var>,

    // output = activation(dot(input, kernel) + bias)

    // units: usize (dimentionality of output space)
    // activation
    // use_bias
    // kernel_initializer
    // bias_initializer
    // kernel_regularizer
    // bias_regularizer
    // activity_regularizer
    // kernel_constraint
    // bias_constraint
}

impl Layer<Tensor, Tensor> for Linear {
    
}

#[derive_opt(LinearOpt)]
#[derive(Default)]
pub struct LinearArg {
    use_bias: Option<bool>,
}
