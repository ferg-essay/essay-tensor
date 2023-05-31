use std::marker::PhantomData;

use crate::{layer::{BoxLayer, Layer}, prelude::Tensors};

pub struct Sequential<O>
where
    O: Tensors
{
    _name: Option<String>,
    layers: Vec<BoxLayer>,
    marker: PhantomData<O>,
}

impl<O> Sequential<O>
where
    O: Tensors
{
    pub fn add<O2: Tensors>(self, into_layer: impl IntoLayer<O, O2>) -> Sequential<O2> {
        let mut layers = self.layers;
        layers.push(into_layer.into_layer());

        Sequential {
            _name: self._name,
            layers,

            marker: PhantomData,
        }
    }
}

impl<L:Layer + 'static> From<L> for BoxLayer {
    fn from(value: L) -> Self {
        Box::new(value)
    }
}

pub trait IntoLayer<O: Tensors, O2: Tensors> {
    fn into_layer(self) -> BoxLayer;
}