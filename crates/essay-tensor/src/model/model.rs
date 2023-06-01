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

impl<I: Tensors<Item=I>, O: Tensors<Item=O>> Model<I, O> {
    pub(crate) fn new(fun: Function<I, O>) -> Self
    {
        Self {
            expr: fun,
        }
    }

    pub fn call(&mut self, input: I) -> O {
        self.expr.call(input)
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

pub enum CallMode {
    Eval,
    Train,
}

pub struct NameScope;

//impl<I: Tensors, O: Tensors, L: Layer> From<L> Model<I, O> {
//
//}

pub struct ModelBuilder<I: Tensors, O: Tensors> {
    id: ModelId,
    inner: ModelInner,
    input: I::ModelIn,
    marker: PhantomData<(I, O)>,
}

/// ModelBuilder is created with sample input
pub fn model_builder<In: Tensors>(input: In) -> ModelBuilder<In, In> {
    let id = ModelId::alloc();

    /*
    let mut expr = Expr::new(id);

    let mut index = 0;
    // TODO: check that the clone and &Tensor work together
    let mut tensors = expr.tensors().clone();
    let args = In::make_arg(&tensors, &mut index);

    let arg_len = In::push_arg(&mut tensors, 0, &input);

    for arg_id in 0..arg_len {
        let arg_id = TensorId::new(id.index() as u32, arg_id as u32);

        let tensor = tensors.get(arg_id).unwrap().clone();
        expr.arg(tensor);
    }
    */

    let inner = Rc::new(RefCell::new(BuilderInner {
        id: id,
        expr: Some(Expr::new(id)),
        tensors: TensorCache::new(id),
    }));

    let inner = ModelInner(inner);

    let model_input = In::model_in(&inner, &input);

    let builder = ModelBuilder {
        id,
        inner,
        input: model_input,
        marker: PhantomData,
    };

    builder
}

impl<I: Tensors<Item=I>, O: Tensors> ModelBuilder<I, O> {
    pub fn add_layer<O1: Tensors>(
        &mut self,
        builder: impl FnMut(O, CallMode) -> O1
    ) -> ModelBuilder<I, O1> {
        todo!()
    }

    pub(crate) fn shape(&self) -> Shape {
        todo!()
    }

    pub(crate) fn input(&self) -> &I::ModelIn {
        &self.input
    }

    pub(crate) fn output<Out: Tensors<Item=Out>>(&self, out: &Out::ModelIn) -> Model<I, Out> {
        // let out = fun(args);

        let mut out_ids : Vec<TensorId> = Vec::new();
        Out::model_out(&mut out_ids, &out);

        let expr = self.inner.expr_clone();

        //let fun = Function::<I, Out>::new(expr, out_ids);
        /*
        Self {
            _vars: Default::default(),
            fun: Box::new(move |graph: &Expr, input, fwd_tensors| { 
                let mut out = graph.tensors().clone();

                In::set_arg(&mut out, 0, &input);

                graph.apply(&mut out, fwd_tensors);

                let mut index = 0;
                let value = Out::make_out(&out, &out_ids, &mut index);

                value
            }),

            program: tape.take_graph().unwrap(),
        }
        */

        //Model::new(fun)
        todo!()
    }
}

#[derive(Clone)]
pub struct ModelInner(Rc<RefCell<BuilderInner>>);

impl ModelInner {
    pub(crate) fn arg(&self, tensor: &Tensor) -> ModelIn {
        let id = self.0.borrow_mut().arg(tensor);

        ModelIn::new(id, &self)
    }

    fn shape(&self, id: TensorId) -> Shape {
        self.0.borrow().shape(id).clone()
    }

    fn tensor(&self, id: TensorId) -> Tensor {
        self.0.borrow().tensor(id).clone()
    }

    fn expr_clone(&self) -> Expr {
        self.0.borrow_mut().take_expr()
    }
}
/*
impl Clone for ModelInner {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
*/

pub(crate) struct BuilderInner {
    id: ModelId,
    expr: Option<Expr>,
    tensors: TensorCache,
}

impl BuilderInner {
    fn arg(&mut self, tensor: &Tensor) -> TensorId {
        match &mut self.expr {
            Some(expr) => {
                let id = expr.arg(tensor.clone());

                let index = self.tensors.push(Some(tensor.clone()));
                assert_eq!(id.index(), index);
        
                id
            }
            None => panic!("expression alreay taken")
        }
    }

    fn shape(&self, id: TensorId) -> &Shape {
        match self.tensors.get(id) {
            Some(tensor) => tensor.shape(),
            None => panic!("Tensor is not set for shape"),
        }
    }

    fn tensor(&self, id: TensorId) -> &Tensor {
        match self.tensors.get(id) {
            Some(tensor) => tensor,
            None => panic!("Tensor is not set for shape"),
        }
    }

    fn take_expr(&mut self) -> Expr {
        self.expr.take().unwrap()
    }
}

pub struct ModelIn<T=f32> {
    id: TensorId,
    builder: ModelInner,
    marker: PhantomData<T>,
}

impl ModelIn {
    fn new(id: TensorId, builder: &ModelInner) -> Self {
        Self {
            id,
            builder: builder.clone(),
            marker: PhantomData,
        }
    }

    pub fn id(&self) -> TensorId {
        self.id
    }

    pub fn shape(&self) -> Shape {
        self.builder.shape(self.id)
    }

    pub fn tensor(&self) -> Tensor {
        self.builder.tensor(self.id)
    }

    fn build<'a, I: ModelsIn, O: ModelsIn>(
        input: I::In<'a>,
        fun: impl FnMut(I::Tin<'_>) -> O::Tout
    ) -> O::Out {
        todo!();
    }
}

impl fmt::Debug for ModelIn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelIn")
            .field("id", &self.id)
            .finish()
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
}
