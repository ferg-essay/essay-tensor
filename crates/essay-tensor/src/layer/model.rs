use crate::{prelude::{Tensors, Shape}, function::Var};

use super::Layer;

pub struct Model<I: Tensors, O: Tensors> {
    fun: Box<dyn FnMut(I, CallMode) -> O>,

    variables: Vec<Var>,
    name_scope: NameScope,

    // layers
    // metrics_names
    // run_eagerly

    // sub_models: Vec<Box<ModelTrait>>,

    // ().training(bool).mask(Tensor<bool>)

    // compile(
    //    optimizer,
    //    metrics,
    //    loss_weights,
    //    run_eagerly,
    //    steps_per_execution,
    // )

    // compute_loss(
    //   x, y, y_pred, sample_weight
    // )

    // compute_metrics(
    //   x, y, y_pred, sample_weight
    //

    // evaluate(
    //   x, y,
    //   batch_size,
    //   verbose,
    //   sample_weight,
    //   steps,
    //   callbacks,
    //   max_queue_size,
    //   workers,
    //   use_multiprocessing
    // )

    // fit(
    //   x, y,
    //   batch_size,
    //   verbose,
    //   sample_weight,
    //   steps,
    //   callbacks,
    //   max_queue_size,
    //   shuffle,
    //   class_weight,
    //   initial_epoch,
    //   steps_per_epoch,
    //   validation_split,
    //   validation_data,
    //   workers,
    //   use_multiprocessing
    // )

    //
    // get_layer(
    //   name,
    //   index,
    // )

    // get_metrics_result()

    // get_weight_paths() - retrieve dictionary of all variables and paths

    // summary(
    //   line_length,
    //   positions,
    //   print_fn,
    //
    // test_on_batch(
    //   x, y,
    //   sample_weight,
    //   reset_metrics
    // )
    //
    // test_step(data)
    //
    // train_on_batch(
    //   x, y,
    //   sample_weight,
    //   class_weight,
    //   reset_metrics
    // )
    //
    // train_step(data)

}

pub enum CallMode {
    Eval,
    Train,
}

impl<I: Tensors, O: Tensors> Model<I, O> {
    pub fn call(&mut self, input: I) -> O {
        (self.fun)(input, CallMode::Eval)
    }

    pub fn build(layers: impl Layer<I, O>) -> Self {
        todo!()
    }
    //
    // reset_metrics
    // compute_metrics()
    //
    // evaluate()
    // fit()
    //
    // get_layer(name, index)
    // get_metrics_result
    // get_weight_paths() - variables
    // load_weights
    // save_weights
    // make_predict_function
    // make_test_function
    // make_train_function
    //
    // predict
    // predict_step
    // predict_on_batch
    //
    // reset_states
    // summary()
    //
    // test_on_batch
    // test_step
    //
    // train_on_batch
    // train_step
}

pub struct NameScope;

//impl<I: Tensors, O: Tensors, L: Layer> From<L> Model<I, O> {
//
//}

pub struct ModelBuilder<I: Tensors, O: Tensors> {
    fun: Box<dyn FnMut(I, CallMode) -> O>,

    variables: Vec<Var>,
    name_scope: NameScope,
}

impl<I: Tensors, O: Tensors> ModelBuilder<I, O> {
    pub fn new() -> Self {
        todo!()
    }

    pub fn add_layer<O1: Tensors>(
        &mut self,
        builder: impl FnMut(O, CallMode) -> O1
    ) -> ModelBuilder<I, O1> {
        todo!()
    }

    pub(crate) fn shape(&self) -> Shape {
        todo!()
    }

    pub(crate) fn input(&self) -> super::layer::LayerIn {
        todo!()
    }

    pub(crate) fn output(&self, mb_out: &super::layer::LayerIn) -> bool {
        todo!()
    }
}

pub trait ModelIn<O: Tensors>  {
}
/*
impl<I, O> ModelIn<O> for ModelBuilder<I, O> 
where
    I: Tensors + 'static,
    O: Tensors + 'static,
{
    type Inputs<'a> = &'a ModelBuilder<I, O>;
    type Outputs = ModelBuilder<I, O>;
}

impl<I, O1, O2, M1, M2> ModelIn<(O1, O2)> for (M1, M2)
where
    I: Tensors + 'static,
    O1: Tensors + 'static,
    O2: Tensors + 'static,
    M1: ModelIn<O1> + 'static,
    M2: ModelIn<O2> + 'static,
{
    type Inputs<'a> = (M1::Inputs<'a>, M2::Inputs<'a>);
    type Outputs = (M1::Outputs, M2::Outputs);
}
*/

