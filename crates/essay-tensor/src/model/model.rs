use std::{marker::PhantomData, sync::atomic::{AtomicU32, Ordering}};

use crate::{prelude::{Shape}, model::Var, Tensor, tensor::Dtype, layer::Layer};

use super::Tensors;

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

/// ModelBuilder is created with sample input
pub fn model_builder<In: Tensors<Item=In>>(input: In) -> ModelBuilder<In, In> {
    todo!()
}

pub struct ModelBuilder<I: Tensors, O: Tensors> {
    fun: Box<dyn FnMut(I, CallMode) -> O>,

    variables: Vec<Var>,
    name_scope: NameScope,
}

impl<I: Tensors, O: Tensors> ModelBuilder<I, O> {
    pub fn add_layer<O1: Tensors>(
        &mut self,
        builder: impl FnMut(O, CallMode) -> O1
    ) -> ModelBuilder<I, O1> {
        todo!()
    }

    pub(crate) fn shape(&self) -> Shape {
        todo!()
    }

    pub(crate) fn input(&self) -> ModelIn {
        todo!()
    }

    pub(crate) fn output<M: ModelsIn>(&self, mb_out: M) -> Model<I, M::Tout> {
        todo!()
    }
}

pub struct ModelIn<T=f32> {
    marker: PhantomData<T>,
}

impl ModelIn {
    pub(crate) fn shape(&self) -> Shape {
        todo!()
    }

    fn build<'a, I: ModelsIn, O: ModelsIn>(
        input: I::In<'a>,
        fun: impl FnMut(I::Tin<'_>) -> O::Tout
    ) -> O::Out {
        todo!();
    }
}

pub trait ModelsIn {
    type In<'a>;
    type Out;
    type Tin<'a>;
    type Tout : Tensors<Item=Self::Tout>;

    fn build<'a, O: ModelsIn>(
        input: Self::In<'a>,
        fun: impl FnMut(Self::Tin<'a>) -> O::Tout
    ) -> O::Out;
}

impl ModelsIn for ModelIn<f32> {
    type In<'a> = &'a ModelIn<f32>;
    type Out = ModelIn<f32>;
    type Tin<'a> =&'a Tensor<f32>;
    type Tout = Tensor<f32>;

    fn build<'a, O: ModelsIn>(
        input: Self::In<'a>,
        fun: impl FnMut(Self::Tin<'a>) -> O::Tout
    ) -> O::Out {
        todo!()
    }
}

impl<L1, L2> ModelsIn for (L1, L2)
where
    L1: ModelsIn,
    L2: ModelsIn,
{
    type In<'a> = (L1::In<'a>, L2::In<'a>);
    type Out = (L1::Out, L2::Out);
    type Tin<'a> = (L1::Tin<'a>, L2::Tin<'a>);
    type Tout = (L1::Tout, L2::Tout);

    fn build<'a, O: ModelsIn>(
        input: Self::In<'a>,
        fun: impl FnMut(Self::Tin<'a>) -> O::Tout
    ) -> O::Out {
        todo!()
    }
}

impl<L> ModelsIn for Vec<L>
where
    L: ModelsIn,
{
    type In<'a> = Vec<L::In<'a>>;
    type Out = Vec<L::Out>;
    type Tin<'a> = Vec<L::Tin<'a>>;
    type Tout = Vec<L::Tout>;

    fn build<'a, O: ModelsIn>(
        input: Self::In<'a>,
        fun: impl FnMut(Self::Tin<'a>) -> O::Tout
    ) -> O::Out {
        todo!()
    }
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

static MODEL_ID: AtomicU32 = AtomicU32::new(0);

///
/// VarId is globally unique to avoid name collisions.
///
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ModelId(u32);

impl ModelId {
    pub(crate) fn alloc() -> ModelId {
        let id = MODEL_ID.fetch_add(1, Ordering::SeqCst);

        ModelId(id)
    }

    #[inline]
    pub fn index(&self) -> usize {
        self.0 as usize
    }
}


#[cfg(test)]
mod test {
    use crate::{Tensor, prelude::Shape, model::{ModelBuilder, ModelIn}, layer::LayerBuilder};

    use super::model_builder;

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

        let model = mb.output(mb_out);

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
