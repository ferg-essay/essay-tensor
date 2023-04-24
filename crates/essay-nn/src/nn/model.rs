use super::Tensor;

pub trait Model {
    fn n_input(&self) -> usize;
    fn n_output(&self) -> usize;

    fn forward(&mut self, data: Tensor<1>) -> Tensor<1>;

    fn box_clone(&self) -> Box<dyn Model>;
}