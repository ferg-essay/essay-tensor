use core::fmt;

use ndarray::Array2;

use super::Tensor;

pub trait Model : fmt::Debug {
    fn n_input(&self) -> usize;
    fn n_output(&self) -> usize;

    fn forward(&mut self, data: &Array2<f32>) -> Array2<f32>;

    fn box_clone(&self) -> Box<dyn Model>;
}