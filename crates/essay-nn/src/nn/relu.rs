use super::Model;

#[derive(Clone, Debug)]
pub struct ReLU {
    len: usize,
}

impl ReLU {
    pub fn new(n: usize) -> Box<dyn Model> {
        Box::new(Self {
            len: n,
        })
    }
}

impl Model for ReLU {
    fn n_input(&self) -> usize {
        self.len
    }

    fn n_output(&self) -> usize {
        self.len
    }

    fn forward(&mut self, data: super::Tensor<1>) -> super::Tensor<1> {
        todo!()
    }

    fn box_clone(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }
}