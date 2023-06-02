use std::marker::PhantomData;

use essay_opt::derive_opt;

use crate::{prelude::{Shape}, flow::FlowIn, Tensor, tensor::Dtype, model::{CallMode, Tensors, ModelContext}};

pub trait Layer<I: Tensors=Tensor, O: Tensors=Tensor> {
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

pub trait LayerBuilder<I: Tensors=Tensor, O: Tensors=Tensor> {
    fn build(&self, input: &I, ctx: &mut ModelContext) -> O;
}


pub trait LayerTrait {

}

#[derive_opt(LayerOpt)]
#[derive(Default)]
pub struct LayerArg {

}

pub type BoxLayer = Box<dyn LayerTrait>;

#[cfg(test)]
mod test {
    use crate::{Tensor, model::{Function, ModelContext}};

    use super::{LayerBuilder};

    #[test]
    fn test() {
        let fun = Function::new(Tensor::zeros([8]), |x, ctx| {
            let (a, b) = Split.build(&x, ctx);

            let a = Plain.build(&a, ctx);
            let b = Plain.build(&b, ctx);

            Sum.build(&vec![a, b], ctx)
        });
    }

    pub struct Split;

    impl LayerBuilder<Tensor, (Tensor, Tensor)> for Split {
        fn build(&self, x: &Tensor, ctx: &mut ModelContext) -> (Tensor, Tensor) {
            let v = ctx.with_layer(self, x, |x, ctx| {
                (x.clone(), x.clone())
            });

            (x + v.0, x + v.1)
        }
    }

    pub struct Plain;

    impl LayerBuilder for Plain {
        fn build(&self, x: &Tensor, ctx: &mut ModelContext) -> Tensor {
            x.clone()
        }
    }

    pub struct Sum;

    impl LayerBuilder<Vec<Tensor>, Tensor> for Sum {
        fn build(&self, x: &Vec<Tensor>, ctx: &mut ModelContext) -> Tensor {
            x[0].clone()
        }
    }
}
