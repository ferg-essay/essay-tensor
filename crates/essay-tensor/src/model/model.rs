use crate::{Tensor, layer::Layer, tensor::Bundle};

pub trait Model<In:Bundle<Item=In>,Out:Bundle<Item=Out>> {
    fn call(&self, input: In) -> Out;

    // fn compile(&self) -> CompileBuilder;

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
