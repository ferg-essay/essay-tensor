use std::marker::PhantomData;

use crate::{prelude::{Tensors, Shape}, flow::FlowIn, Tensor, tensor::Dtype};

use super::model::{CallMode, Model, ModelIn, ModelBuilder};

pub trait Layer<I: Tensors, O: Tensors> {
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


pub trait LayerBuilder<I: LayersIn, O: LayersIn> {
    fn build<'a>(&self, input: I::In<'a>) -> O::Out;

    fn build_model<'a>(
        input: I::In<'a>,
        fun: impl FnMut(I::Tin<'a>) -> O::Tout
    ) -> O::Out {
        todo!()
    }
    // fn compile<In: Tensors>(&self, input: &mut ModelBuilder<In, I>) -> ModelBuilder<In, O>;
}

pub struct LayerIn<T=f32> {
    marker: PhantomData<T>,
}
impl LayerIn {
    pub(crate) fn shape(&self) -> Shape {
        todo!()
    }

    fn build<'a, I: LayersIn, O: LayersIn>(
        input: I::In<'a>,
        fun: impl FnMut(I::Tin<'_>) -> O::Tout
    ) -> O::Out {
        todo!();
    }
}

pub trait LayersIn {
    type In<'a>;
    type Out;
    type Tin<'a>;
    type Tout;

    fn build<'a, O: LayersIn>(
        input: Self::In<'a>,
        fun: impl FnMut(Self::Tin<'a>) -> O::Tout
    ) -> O::Out;
}

impl<T: Dtype> LayersIn for LayerIn<T> {
    type In<'a> = &'a LayerIn<T>;
    type Out = LayerIn<T>;
    type Tin<'a> =&'a Tensor<T>;
    type Tout = Tensor<T>;

    fn build<'a, O: LayersIn>(
        input: Self::In<'a>,
        fun: impl FnMut(Self::Tin<'a>) -> O::Tout
    ) -> O::Out {
        todo!()
    }
}

impl<L1, L2> LayersIn for (L1, L2)
where
    L1: LayersIn,
    L2: LayersIn,
{
    type In<'a> = (L1::In<'a>, L2::In<'a>);
    type Out = (L1::Out, L2::Out);
    type Tin<'a> = (L1::Tin<'a>, L2::Tin<'a>);
    type Tout = (L1::Tout, L2::Tout);

    fn build<'a, O: LayersIn>(
        input: Self::In<'a>,
        fun: impl FnMut(Self::Tin<'a>) -> O::Tout
    ) -> O::Out {
        todo!()
    }
}

impl<L> LayersIn for Vec<L>
where
    L: LayersIn,
{
    type In<'a> = Vec<L::In<'a>>;
    type Out = Vec<L::Out>;
    type Tin<'a> = Vec<L::Tin<'a>>;
    type Tout = Vec<L::Tout>;

    fn build<'a, O: LayersIn>(
        input: Self::In<'a>,
        fun: impl FnMut(Self::Tin<'a>) -> O::Tout
    ) -> O::Out {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use crate::{Tensor, layer::model::ModelBuilder};

    use super::{LayerIn, LayerBuilder, LayersIn};

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

    impl LayerBuilder<LayerIn, (LayerIn, LayerIn)> for Split {
        fn build(&self, input: &LayerIn) -> (LayerIn, LayerIn) {
            Self::build_model(input, |x| { (x.clone(), x.clone()) })
            /*
            LayerIn::build::<LayerIn, (LayerIn, LayerIn)>(
                input,
                |x| { (x, x) })
                */
        }
    }

    pub struct Plain;

    impl LayerBuilder<LayerIn, LayerIn> for Plain {
        fn build(&self, input: &LayerIn) -> LayerIn {
            Self::build_model(input, |x| x.clone())
            /*
            LayerIn::build::<LayerIn, (LayerIn, LayerIn)>(
                input,
                |x| { (x, x) })
                */
        }
    }

    pub struct Sum;

    impl LayerBuilder<Vec<LayerIn>, LayerIn> for Sum {
        fn build(&self, input: Vec<&LayerIn>) -> LayerIn {
            Self::build_model(input, |x: Vec<&Tensor>| {
                x[0].clone()
            })
        }
    }
}
