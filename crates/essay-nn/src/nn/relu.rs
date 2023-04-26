use ndarray::Array2;

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

    fn forward(&mut self, data: &Array2<f32>) -> Array2<f32> {
        let mut data = data.clone();
        
        for item in &mut data.iter_mut() {
            *item = if *item > 0. { *item } else { 0. };
        }

        data
    }

    fn box_clone(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }
}