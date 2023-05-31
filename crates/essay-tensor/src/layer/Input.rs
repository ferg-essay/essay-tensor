use crate::{prelude::Shape, Tensor};

pub struct InputLayer {
    input_shape: InputSpec, // Shape pattern
    batch_size: Option<usize>,
    input_tensor: Option<Tensor>,
    name: Option<String>,
}

pub struct InputSpec {
    shape: Vec<Option<usize>>,
    // shape with possible None values

    ndim: usize,
    max_ndim: usize,
    min_ndim: usize,
    // axes
    // allow_last_axis_squeeze
}