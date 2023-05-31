use crate::prelude::Tensors;

pub trait Layer<I, O>
where
    I: Tensors<Item=I>,
    O: Tensors<Item=O>,
{
    // trainable: bool
    // name
    // trainable_weights -> Var
    // non_trainable_weights -> Var
    // weights
    // input_spec
    // activity_regularizer,
    // input -- input tensor (if one input)
    // losses
    // metrics
    // output - (only if one output)
    //
    // add_loss(losses)
    //
    // add_weight(
    //   name
    //   shape, - default to scalar
    //   initializer,
    //   regularizer,
    //   trainable,
    //   constraint,
    // )
    //
    // build(input_shape)
    //
    // call(inputs, args) - args additional positional arguments
    //
    // compute_output_shape(input_shape)
    // compute_output_signature(input_signature)
    // count_params()
    //
    // get_weights()
    // set_weights()
    //
    // __call__
}

pub type BoxLayer<I, O> = Box<dyn Layer<I, O>>;