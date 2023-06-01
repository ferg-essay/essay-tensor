use core::fmt;
use std::{marker::PhantomData, sync::atomic::{AtomicU32, Ordering}, rc::Rc, cell::RefCell};

use crate::{prelude::{Shape}, model::{Var, Function}, Tensor, layer::Layer, tensor::TensorId};

use super::{Tensors, Expr, TensorCache};

pub struct Model<I: Tensors, O: Tensors> {
    expr: Function<I, O>,

    //variables: Vec<Var>,
    //name_scope: NameScope,

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

impl<I: Tensors, O: Tensors> Model<I, O> {
    pub(crate) fn new(fun: Function<I, O>) -> Self
    {
        Self {
            expr: fun,
        }
    }

    pub fn call(&mut self, input: I) -> O {
        self.expr.call(input)
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

pub enum CallMode {
    Eval,
    Train,
}

pub struct NameScope;


#[cfg(test)]
mod test {
    /*
    use crate::{Tensor, prelude::Shape, model::{ModelBuilder, ModelIn}, layer::LayerBuilder, tensor::TensorId, init::random_normal};

    use super::model_builder;

    #[test]
    fn model_builder_single_input() {
        let mb = model_builder(Tensor::zeros([8]));
        let input = mb.input();

        assert_eq!(input.id(), TensorId::new(0, 0));
        assert_eq!(input.shape(), Shape::from([8]));
        assert_eq!(input.tensor(), Tensor::zeros([8]));
    }

    #[test]
    fn model_builder_identity() {
        let mb = model_builder(Tensor::zeros([8]));
        let input = mb.input();
        let mut model = mb.output::<Tensor>(input);

        assert_eq!(model.call(Tensor::zeros([8])), Tensor::zeros([8]));
        assert_eq!(model.call(Tensor::ones([8])), Tensor::ones([8]));

        let values = random_normal([8], ());
        assert_eq!(model.call(values.clone()), values);
    }

    #[test]
    fn test_layer() {
        let mb = model_builder(Tensor::zeros([8]));
        let input = mb.input();

        let l = Split;

        let (a, b) = l.build(&input);

        let la = Plain;
        let lb = Plain;

        let a = la.build(&a);
        let b = lb.build(&b);

        let lsum = Sum;

        let mb_out = lsum.build(vec![&a, &b]);

        let model = mb.output::<Tensor>(&mb_out);

    }

    pub struct Split;

    impl LayerBuilder<ModelIn, (ModelIn, ModelIn)> for Split {
        fn build(&self, input: &ModelIn) -> (ModelIn, ModelIn) {
            Self::build_model(input, |x| { (x.clone(), x.clone()) })
        }
    }

    pub struct Plain;

    impl LayerBuilder<ModelIn, ModelIn> for Plain {
        fn build(&self, input: &ModelIn) -> ModelIn {
            Self::build_model(input, |x| x.clone())
        }
    }

    pub struct Sum;

    impl LayerBuilder<Vec<ModelIn>, ModelIn> for Sum {
        fn build(&self, input: Vec<&ModelIn>) -> ModelIn {
            Self::build_model(input, |x: Vec<&Tensor>| {
                x[0].clone()
            })
        }
    }
    */
}
