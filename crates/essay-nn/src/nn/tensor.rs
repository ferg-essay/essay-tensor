#[derive(Clone, Debug)]
pub struct Tensor<const N:usize> {
    shape: [usize; N],
    data: Vec<f32>,
}