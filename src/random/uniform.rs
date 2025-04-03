use rand::prelude::*;

use crate::{tensor::Shape, Tensor};

pub fn uniform(
    shape: impl Into<Shape>,
    min: f32, 
    max: f32, 
    seed: Option<u64>,
) -> Tensor {
    match seed {
        Some(seed) => {
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

            Tensor::init(shape, || rng.gen_range(min..max))
        }
        None => { 
            let mut rng = thread_rng() ;

            Tensor::init(shape, || rng.gen_range(min..max))
        }
    }
}

pub fn uniform_b(
    shape: impl Into<Shape>,
) -> UniformBuilder {
    UniformBuilder::new(shape)
}

pub struct UniformBuilder {
    shape: Shape,
    min: f32,
    max: f32,
    seed: Option<u64>,
}

impl UniformBuilder {
    fn new(shape: impl Into<Shape>) -> Self {
        Self {
            shape: Into::into(shape),
            min: 0.,
            max: 1.,
            seed: None,
        }
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);

        self
    }

    pub fn go(self) -> Tensor {
        uniform(self.shape, self.min, self.max, self.seed)
    }
}

#[cfg(test)]
mod test {
    use super::uniform_b;
    use crate::prelude::*;

    #[test]
    fn test_one() {
        let t = uniform_b([1]).seed(100).go();
        assert_eq!(t, ten!([0.9268179]));
    }

    #[test]
    fn test_vector() {
        let t = uniform_b([3]).seed(100).go();
        assert_eq!(t, ten!([0.9268179, 0.55682373, 0.57285655]));
    }

    #[test]
    fn test_matrix() {
        let t = uniform_b([3, 2]).seed(100).go();
        assert_eq!(t, ten!([
            [0.9268179, 0.55682373, 0.57285655],
            [0.68730843, 0.8088945, 0.3300841]
        ]));
    }
}

