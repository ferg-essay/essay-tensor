use crate::tensor::Tensor;

pub fn arange(start: f32, end: f32, step: f32) -> Tensor {
    assert!(step != 0.);

    let mut start = start;

    if start <= end && step > 0. {
        let mut vec = Vec::<f32>::new();

        while start < end {
            vec.push(start);
            start += step;
        }

        Tensor::from(vec)
    } else if end <= start && step < 0. {
        let mut vec = Vec::<f32>::new();

        while end < start {
            vec.push(start);
            start -= step;
        }

        Tensor::from(vec)
    } else {
        panic!("invalid arguments start={} end={} step={}", start, end, step);
    }
}

impl Tensor {
    pub fn arange(start: f32, end: f32, step: f32) -> Tensor {
        arange(start, end, step)
    }
}
