use crate::layer::{BoxLayer, Layer};

pub struct Sequential {
    _name: Option<String>,
    layers: Vec<BoxLayer>,
}

impl Sequential {
    pub fn add(&mut self, into_layer: impl Into<BoxLayer>) -> &mut Self {
        self.layers.push(into_layer.into());
        self
    }
}

impl<L:Layer + 'static> From<L> for BoxLayer {
    fn from(value: L) -> Self {
        Box::new(value)
    }
}