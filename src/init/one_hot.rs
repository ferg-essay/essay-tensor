use crate::Tensor;

pub fn one_hot(indices: impl AsRef<[usize]>, depth: usize) -> Tensor {
    let mut vec = Vec::<f32>::new();
    vec.resize(depth, 0.);

    for i in indices.as_ref() {
        vec[*i] = 1.;
    }

    Tensor::from(vec)
}

impl Tensor {
    pub fn one_hot(indices: impl AsRef<[usize]>, depth: usize) -> Tensor {
        one_hot(indices, depth)
    }
}

#[cfg(test)]
mod test {
    use crate::{init::one_hot, tf32};

    #[test]
    fn basic_one_hot() {
        assert_eq!(one_hot([0], 3), tf32!([1., 0., 0.]));
        assert_eq!(one_hot([1], 3), tf32!([0., 1., 0.]));
        assert_eq!(one_hot(vec![2], 3), tf32!([0., 0., 1.]));
        assert_eq!(one_hot(vec![0, 2], 3), tf32!([1., 0., 1.]));
    }
}
