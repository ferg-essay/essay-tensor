use crate::random::Rand32;
use crate::{Tensor, prelude::Shape};
use crate::ops::{init_op, InitKernel};

pub fn random_uniform(
    shape: impl Into<Shape>,
    opt: impl UniformOpt<RandomUniform>,
) -> Tensor {
    init_op(opt.into_arg(), shape)
}

pub fn random_uniform_initializer(
    opt: impl UniformOpt<RandomUniform>,
) -> RandomUniform {
    opt.into_arg()
}

impl Tensor {
    pub fn random_uniform(shape: impl Into<Shape>, opt: impl UniformOpt<RandomUniform>) -> Tensor {
        random_uniform(shape, opt)
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct RandomUniform {
    min: f32,
    max: f32,
    seed: Option<u64>,
}

impl InitKernel<f32> for RandomUniform {
    type State = Rand32;

    fn init(&self, _shape: &Shape) -> Self::State {
        match self.seed {
            Some(seed) => Rand32(seed),
            None => Rand32::new(),
        }
    }

    fn f(&self, state: &mut Self::State) -> f32 {
        let rng = state.next_uniform();

        self.min + (self.max - self.min) * rng as f32
    }
}

impl Default for RandomUniform {
    fn default() -> Self {
        Self { 
            min: 0.,
            max: 1.,
            seed: None,
        }
    }
}

pub trait UniformOpt<Arg> {
    fn into_arg(self) -> Arg;

    fn min(self, min: f32) -> Arg;
    fn max(self, max: f32) -> Arg;
    fn seed(self, seed: u64) -> Arg;
}

impl UniformOpt<RandomUniform> for RandomUniform {
    fn into_arg(self) -> Self {
        self
    }

    fn min(self, min: f32) -> Self {
        Self { min, ..self }
    }

    fn max(self, min: f32) -> Self {
        Self { min, ..self }
    }

    fn seed(self, seed: u64) -> Self {
        Self { seed: Some(seed), ..self }
    }
}

impl UniformOpt<RandomUniform> for () {
    fn into_arg(self) -> RandomUniform {
        RandomUniform::default()
    }

    fn min(self, min: f32) -> RandomUniform {
        self.into_arg().min(min)
    }

    fn max(self, max: f32) -> RandomUniform {
        self.into_arg().max(max)
    }

    fn seed(self, seed: u64) -> RandomUniform {
        self.into_arg().seed(seed)
    }
}


#[cfg(test)]
mod test {
    use random_uniform::UniformOpt;

    use crate::prelude::*;
    use crate::init::random_uniform;

    #[test]
    fn test_uniform() {
        assert_eq!(
            random_uniform([1], ().seed(100)), 
            tf32!([0.5568238])
        );

        // check seed is consistent
        assert_eq!(
            random_uniform([1], ().seed(100)), 
            tf32!([0.5568238])
        );
        
        assert_eq!(
            random_uniform([4], ().seed(100)), 
            tf32!([0.5568238, 0.68730855, 0.33008412, 0.96558535])
        );
        
        assert_eq!(
            random_uniform([2, 2], ().seed(100)), 
            tf32!([[0.5568238, 0.68730855], [0.33008412, 0.96558535]])
        );
        
        assert_eq!(
            random_uniform([3, 2, 2], ().seed(100)), 
            tf32!([
                [[0.5568238, 0.68730855], [0.33008412, 0.96558535]],
                [[0.7637636, 0.20205542], [0.9115083, 0.68375206]],
                [[0.11434494, 0.29887143], [0.027762651, 0.66628486]]            
            ])
        );
    }
}
