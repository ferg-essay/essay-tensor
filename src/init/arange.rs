use crate::{
    Tensor, 
    tensor::{TensorVec}, 
};

pub fn arange(start: f32, end: f32, step: f32) -> Tensor {
    assert!(step != 0.);

    let mut start = start;

    if start <= end && step > 0. {
        let mut vec = TensorVec::<f32>::new();

        while start < end {
            vec.push(start);
            start += step;
        }

        vec.into_tensor()
    } else if end <= start && step < 0. {
        let mut vec = TensorVec::<f32>::new();

        while end < start {
            vec.push(start);
            start += step;
        }

        vec.into_tensor()
    } else {
        panic!("invalid arguments start={} end={} step={}", start, end, step);
    }
}

impl Tensor {
    pub fn arange(start: f32, end: f32, step: f32) -> Tensor {
        arange(start, end, step)
    }
}
