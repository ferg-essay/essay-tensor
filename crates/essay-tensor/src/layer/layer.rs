use std::marker::PhantomData;

use crate::{prelude::{Tensors, Shape}, flow::FlowIn, Tensor, tensor::Dtype, model::{CallMode, ModelsIn, ModelIn}};

pub trait Layer<I: Tensors = Tensor, O: Tensors = Tensor> {
    fn call(&self, input: I, mode: CallMode) -> O;

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

pub trait LayerTrait {

}

pub type BoxLayer = Box<dyn LayerTrait>;


pub trait LayerBuilder<I: ModelsIn = ModelIn, O: ModelsIn = ModelIn> {
    fn build<'a>(
        &self, 
        input: I::In<'a>,
    ) -> O::Out;

    fn build_model<'a>(
        input: I::In<'a>,
        fun: impl FnMut(I::Tin<'a>) -> O::Tout
    ) -> O::Out {
        todo!()
    }
    // fn compile<In: Tensors>(&self, input: &mut ModelBuilder<In, I>) -> ModelBuilder<In, O>;
}

#[cfg(test)]
mod test {
    use crate::{Tensor, prelude::Shape, model::{ModelBuilder, ModelIn}};

    use super::{LayerBuilder};

    fn test() {
        let mb = ModelBuilder::<Tensor, Tensor>::new();
        let input = mb.input();

        let l = Split;

        let (a, b) = l.build(&input);

        let la = Plain;
        let lb = Plain;

        let a = la.build(&a);
        let b = lb.build(&b);

        let lsum = Sum;

        let mb_out = lsum.build(vec![&a, &b]);

        let model = mb.output(&mb_out);

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
}
