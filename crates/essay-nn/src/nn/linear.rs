use super::{Tensor, Model};

#[derive(Clone, Debug)]
pub struct Linear {
    shape: [usize; 2],
    weights: Tensor<2>, 
}

impl Linear {
    pub fn new(n_in: usize, n_out: usize) -> Box<dyn Model> {
        todo!();
    }
}

impl Model for Linear {
    fn n_input(&self) -> usize {
        self.shape[0]
    }

    fn n_output(&self) -> usize {
        self.shape[1]
    }

    fn forward(&mut self, data: Tensor<1>) -> Tensor<1> {
        todo!()
    }

    fn box_clone(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }
}