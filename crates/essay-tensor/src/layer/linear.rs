use essay_opt::derive_opt;

use crate::{model::{Var, CallMode, ModelIn}, Tensor, prelude::{Tensors, Shape}, init::{Initializer, zeros_initializer}};

use super::{Layer, input::InputSpec, 
    layer::{LayerBuilder}};

pub struct Linear {
    // var_a : Var,
    // var_b : Option<Var>,

    units: usize,
    init: Box<dyn Initializer>,
    bias_init: Box<dyn Initializer>,

    name: Option<String>,
    use_bias: Option<bool>,

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

impl Linear {
    pub fn new(units: usize) -> Self {
        Self {
            units,
            init: zeros_initializer(),
            bias_init: zeros_initializer(),
            name: None,
            use_bias: None,
        }
    }

    /*
    pub fn build<I: Tensors>(self, model: impl Model<I, Tensor>) -> LinearModel {
        let init : Box<dyn Initializer>;
        let a = Var(init.init([10, 20]), ().name("a"));
        todo!()
    }
    */
}

impl Layer for Linear {
    fn call(&self, input: Tensor, mode: CallMode) -> Tensor {
        todo!()
    }
}

impl LayerBuilder for Linear {
    fn build(&self, input: &ModelIn) -> ModelIn {
        let a = Var::new("a", self.init.init(&input.shape().extend_dims(self.units)));
        let b = Var::new("b",self.bias_init.init(&self.units.into()));

        Self::build_model(input, |x| {
            a.matvec(&x) + &b
        })
    }
}

pub struct LinearModel {
    input_spec: InputSpec,

    var_a: Var,
    var_b: Option<Var>,

    // opt: Linear,
}

#[derive_opt(LinearOpt)]
#[derive(Default)]
pub struct LinearArg {
    name: Option<String>,
    use_bias: Option<bool>,
}
