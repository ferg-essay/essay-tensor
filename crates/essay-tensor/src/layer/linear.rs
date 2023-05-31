use crate::function::Var;

pub struct _Linear {
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