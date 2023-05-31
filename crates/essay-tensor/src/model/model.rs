use std::any::Any;

use crate::{Tensor, layer::Layer, tensor::Tensors, optimizer::Optimizer, loss::Loss};

pub trait Model {
    fn call(&mut self, input: Tensor) -> Tensor;
}

impl<F> Model for F
where
    F: FnMut(Tensor) -> Tensor
{
    fn call(&mut self, input: Tensor) -> Tensor {
        self(input)
    }
}

pub trait ModelTensorflow<In:Tensors<Item=In>, Out:Tensors<Item=Out>> {
    fn call(&self, input: In) -> Out;

    // fn compile(&self) -> CompileBuilder;

    fn compute_loss(&self, x: Option<Tensor>, y: Option<Tensor>, y_pred: Option<Tensor>);

    fn evaluate(&self, x: Option<Tensor>, y: Option<Tensor>);

    // (input, target)
    fn fit(&self, x: Option<Tensor>, y: Option<Tensor>, epochs: usize);

    fn get_layer(&self, name: &str) -> Box<dyn Layer<In, Out>>;

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

pub struct ModelImpl {

}

impl ModelImpl {
    // fn new(input: impl Into<Box<dyn Model>>, output: impl Into<Box<dyn Model>>) -> ModelImpl {
}

pub struct CompileBuilder {
}

impl CompileBuilder {
    fn optimizer(&mut self, opt: impl Into<Box<dyn Optimizer>>) -> &mut Self {
        todo!()
    }

    fn loss(&mut self, loss: impl Into<Box<dyn Loss>>) -> &mut Self {
        todo!()
    }

    fn metrics(&mut self, metrics: impl Into<Box<dyn Any>>) -> &mut Self {
        todo!()
    }

    fn weighted_metrics(&mut self, metrics: impl Into<Box<dyn Any>>) -> &mut Self {
        todo!()
    }

    fn run_eagerly(&mut self, is_eager: bool) -> &mut Self {
        todo!();
    }

    fn steps_per_execution(&mut self, count: u32) -> &mut Self {
        todo!();
    }

    fn compile(self) {
        todo!()
    }
}

pub struct FitBuilder {

}

impl FitBuilder {
    fn x(&mut self, data: Fit<f32>) -> &mut Self {
        todo!()
    }

    fn xy(&mut self, x: Tensor<f32>, y: Tensor<f32>) -> &mut Self {
        todo!()
    }

    fn batch_size(&mut self, count: usize) -> &mut Self {
        // default 32, don't specify for datasets
        // number of samples per gradient update
        todo!()
    }

    fn epochs(&mut self, count: usize) -> &mut Self {
        // each epoch is over the entire data set
        todo!()
    }

    fn verbose(&mut self, verbosity: usize) -> &mut Self {
        todo!()
    }

    fn callbacks(&mut self) -> &mut Self {
        // register callbacks
        todo!()
    }

    fn validation_split(&mut self, fraction: f32) -> &mut Self {
        // only when input is Tensor
        todo!()
    }

    fn validation_data(&mut self, data: (Tensor<f32>, Tensor<f32>)) -> &mut Self {
        // (x_val, y_val)
        todo!()
    }

    fn shuffle(&mut self, is_shuffle: bool) -> &mut Self {
        // if shuffle data before execute (non dataset)
        todo!();
    }
        
    fn steps_per_epoch(&mut self, steps: usize) -> &mut Self {
        // number of batches for an epoch
        todo!();
    }
}

struct Fit<T>(Tensor<T>, Tensor<T>);

impl<T> Fit<T> {
    fn inputs(&self) -> &Tensor<T> { &self.0 }
    fn targets(&self) -> &Tensor<T> { &self.1 }
    // fn sample_weights(&self)
}