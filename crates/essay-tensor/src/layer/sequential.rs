use crate::{Tensor};

use super::LayerBuilder;

pub struct Sequential {
    layers: Vec<BoxLayerOne>,
}

impl Sequential {
    pub fn push(&mut self, layer: impl LayerBuilder<Tensor, Tensor> + 'static) -> &mut Self {
        self.layers.push(Box::new(layer));

        self
    }
}

type BoxLayerOne = Box<dyn LayerBuilder<Tensor, Tensor>>;

impl LayerBuilder<Tensor, Tensor> for Sequential {
    fn build(&self, input: &Tensor, ctx: &mut crate::model::ModelContext) -> Tensor {
        let mut value = input.clone();

        for layer in &self.layers {
            value = layer.build(&value, ctx);
        }

        value
    }
}
