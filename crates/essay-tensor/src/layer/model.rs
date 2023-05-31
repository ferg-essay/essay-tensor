use crate::prelude::Tensors;

pub trait Model<I: Tensors, O: Tensors> {
    // ().training(bool).mask(Tensor<bool>)
    fn call(input: I) -> O;

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