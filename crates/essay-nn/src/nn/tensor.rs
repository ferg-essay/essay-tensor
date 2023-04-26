#[derive(Clone, Debug)]
pub struct Tensor<const N:usize> {
    shape: [usize; N],
    data: Vec<f32>,
}

impl Tensor<1> {
    pub fn new(data: Vec<f32>) -> Self {
        Self {
            shape: [data.len()],
            data,
        }
    }
}
