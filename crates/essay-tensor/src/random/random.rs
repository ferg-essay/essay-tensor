use std::{rc::Rc, sync::Arc};

use rand::prelude::*;

use crate::{tensor::{TensorData, Dtype, TensorUninit}, Tensor};

pub fn uniform<const N:usize>(
    shape: [usize; N], 
    min: f32, 
    max: f32, 
    seed: Option<u64>,
) -> Tensor {
    let len : usize = shape.iter().product();

    unsafe {
        let mut data = TensorUninit::new(len);

        match seed {
            Some(seed) => {
                let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

                for i in 0..len {
                    data[i] = rng.gen_range(min..max);
                }
            }
            None => { 
                let mut rng = thread_rng() ;

                for i in 0..len {
                    data[i] = rng.gen_range(min..max);
                }
            }
        };

        Tensor::new(Arc::new(data.init()), &shape)
    }
}

pub fn uniform_b<const N:usize>(
    shape: [usize; N]
) -> UniformBuilder<N> {
    UniformBuilder::new(shape)
}

pub struct UniformBuilder<const N:usize> {
    shape: [usize; N],
    min: f32,
    max: f32,
    seed: Option<u64>,
}

impl<const N:usize> UniformBuilder<N> {
    fn new(shape: [usize; N]) -> Self {
        Self {
            shape: shape,
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
        assert_eq!(t, tensor!([0.9268179]));
    }

    #[test]
    fn test_vector() {
        let t = uniform_b([3]).seed(100).go();
        assert_eq!(t, tensor!([0.9268179, 0.55682373, 0.57285655]));
    }

    #[test]
    fn test_matrix() {
        let t = uniform_b([3, 2]).seed(100).go();
        assert_eq!(t, tensor!([
            [0.9268179, 0.55682373, 0.57285655],
            [0.68730843, 0.8088945, 0.3300841]
        ]));
    }
}

