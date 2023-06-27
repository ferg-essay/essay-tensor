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

impl InputSpec {
    fn new(vec: Vec<Option<usize>>) -> Self {
        let len = vec.len();

        Self {
            shape: vec,

            ndim: len,
            max_ndim: len,
            min_ndim: len,
        }
    }
}

impl From<usize> for InputSpec {
    fn from(value: usize) -> Self {
        InputSpec::new(vec![Some(value)])
    }
}

impl From<Vec<Option<usize>>> for InputSpec {
    fn from(value: Vec<Option<usize>>) -> Self {
        InputSpec::new(value)
    }
}