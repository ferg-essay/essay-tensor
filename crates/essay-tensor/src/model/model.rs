use crate::{graph::Bundle, Tensor, layer::Layer};

pub trait Model<In:Bundle<Item=In>,Out:Bundle<Item=Out>> {
    fn call(&self, input: In) -> Out;

    fn compile(&self) -> CompileBuilder;

    fn compute_loss(&self, x: Option<Tensor>, y: Option<Tensor>, y_pred: Option<Tensor>);

    fn evaluate(&self, x: Option<Tensor>, y: Option<Tensor>);

    fn fit(&self, x: Option<Tensor>, y: Option<Tensor>, epochs: usize);

    fn get_layer(&self, name: &str) -> Box<dyn Layer>;

    fn get_weight_paths(&self);

    fn make_predict_function(&self);

    fn make_test_function(&self);

    fn make_train_function(&self);

    fn predict(x: Tensor);
    fn predict_step(data: Tensor);

    fn test_step(data: Tensor);

    fn train_on_batch(x: Tensor, y: Option<Tensor>);
    fn train_step(data: Tensor);
}

pub trait Optimizer {}
type BoxOptimizer = Box<dyn Optimizer>;

pub trait Loss {}
type BoxLoss = Box<dyn Loss>;

pub trait Metric {}
type BoxMetric = Box<dyn Metric>;

pub struct CompileBuilder {
}

impl CompileBuilder {
    pub fn optimizer(&mut self, optimizer: impl Into<BoxOptimizer>) -> &mut Self {
        self
    }

    pub fn loss(&mut self, loss: impl Into<BoxLoss>) -> &mut Self {
        self
    }

    pub fn metric(&mut self, metric: impl Into<BoxMetric>) -> &mut Self {
        self
    }

    pub fn metrics(&mut self, metrics: &[impl Into<BoxMetric>]) -> &mut Self {
        self
    }
}