use crate::random::Rand32;
use crate::{Tensor, prelude::Shape};

use super::initializer::Initializer;

pub fn random_normal(
    shape: impl Into<Shape>,
    opt: impl NormalOpt<RandomNormal>,
) -> Tensor {
    // init_op(opt.into_arg(), shape)
    todo!();
}

pub fn random_normal_initializer(
    opt: impl NormalOpt<RandomNormal>,
) -> RandomNormal {
    opt.into_arg()
}

impl Tensor {
    pub fn random_normal(shape: impl Into<Shape>, opt: impl NormalOpt<RandomNormal>) -> Tensor {
        random_normal(shape, opt)
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct RandomNormal {
    mean: f32,
    stddev: f32,
    seed: Option<u64>,
}

//impl InitKernel<f32> for RandomNormal {
/*
impl RandomNormal {
    type State = Rand32;

    fn init(&self, _shape: &Shape) -> Self::State {
        match self.seed {
            Some(seed) => Rand32(seed),
            None => Rand32::new(),
        }
    }

    fn f(&self, state: &mut Self::State) -> f32 {
        let p = state.next_normal();

        self.mean + p * self.stddev
    }
}
*/    

impl Default for RandomNormal {
    fn default() -> Self {
        Self { 
            mean: 0.,
            stddev: 1.,
            seed: None,
        }
    }
}
/*
impl Initializer for RandomNormal {
    fn init(&self, shape: &Shape) -> Tensor {
        init_op(self.clone(), shape)
    }
}
*/    

pub trait NormalOpt<Arg> {
    fn into_arg(self) -> Arg;

    fn mean(self, mean: f32) -> Arg;
    fn stddev(self, stddev: f32) -> Arg;
    fn seed(self, seed: u64) -> Arg;
}

impl NormalOpt<RandomNormal> for RandomNormal {
    fn into_arg(self) -> Self {
        self
    }

    fn mean(self, mean: f32) -> Self {
        Self { mean, ..self }
    }

    fn stddev(self, stddev: f32) -> Self {
        Self { stddev, ..self }
    }

    fn seed(self, seed: u64) -> Self {
        Self { seed: Some(seed), ..self }
    }
}

impl NormalOpt<RandomNormal> for () {
    fn into_arg(self) -> RandomNormal {
        RandomNormal::default()
    }

    fn mean(self, mean: f32) -> RandomNormal {
        self.into_arg().mean(mean)
    }

    fn stddev(self, stddev: f32) -> RandomNormal {
        self.into_arg().stddev(stddev)
    }

    fn seed(self, seed: u64) -> RandomNormal {
        self.into_arg().seed(seed)
    }
}


#[cfg(test)]
mod test {
    use random_normal::NormalOpt;

    use crate::prelude::*;
    use crate::init::random_normal;

    #[test]
    fn test_normal() {
        assert_eq!(
            random_normal([1], ().seed(100)), 
            tf32!([-0.41531694])
        );

        // check seed is consistent
        assert_eq!(
            random_normal([1], ().seed(100)), 
            tf32!([-0.41531694])
        );
        
        assert_eq!(
            random_normal([4], ().seed(100)), 
            tf32!([-0.41531694, 1.4542247, 0.21783249, -0.17405495])
        );
        
        assert_eq!(
            random_normal([2, 2], ().seed(100)), 
            tf32!([[-0.41531694, 1.4542247], [0.21783249, -0.17405495]])
        );
        
        assert_eq!(
            random_normal([3, 2, 2], ().seed(100)), 
            tf32!([
                [[-0.41531694, 1.4542247], [0.21783249, -0.17405495]],
                [[-0.6294868, -1.3442261], [-0.37828407, 1.3928399]],
                [[-0.17792797, -0.11458359], [0.2934363, -0.8526999]],
            ])
        );
    }

    #[test]
    fn test_normal_mean() {
        // base normal distribution
        assert_eq!(
            random_normal([4], ().seed(100)), 
            tf32!([-0.41531694, 1.4542247, 0.21783249, -0.17405495])
        );

        assert_eq!(
            random_normal([4], ().seed(100).mean(100.)), 
            tf32!([99.58469, 101.45422, 100.217834, 99.82594]),
        );

        assert_eq!(
            random_normal([4], ().seed(100).mean(-10.)), 
            tf32!([-10.415317, -8.545775, -9.782167, -10.174055]),
        );
    }

    #[test]
    fn test_normal_stddev() {
        // base normal distribution
        assert_eq!(
            random_normal([4], ().seed(100)), 
            tf32!([-0.41531694, 1.4542247, 0.21783249, -0.17405495])
        );

        assert_eq!(
            random_normal([4], ().seed(100).stddev(10.)), 
            tf32!([-4.1531696, 14.542247, 2.178325, -1.7405496]),
        );

        assert_eq!(
            random_normal([4], ().seed(100).stddev(100.)), 
            tf32!([-41.531693, 145.42247, 21.783249, -17.405495]),
        );
    }

    #[test]
    fn test_normal_reduce_mean() {
        // sanity checking

        assert_eq!(
            random_normal([65536], ().seed(100)).reduce_mean(()),
            tf32!(-0.0054125143)
        );

        assert_eq!(
            random_normal([65536], ().seed(10)).reduce_mean(()),
            tf32!(-0.0005972396)
        );

        assert_eq!(
            random_normal([65536], ().seed(10).mean(2.)).reduce_mean(()),
            tf32!(1.9994035)
        );

        assert_eq!(
            random_normal([65536], ().seed(10).stddev(10.)).reduce_mean(()),
            tf32!(-0.0059724655)
        );
    }


    #[test]
    fn test_normal_reduce_std() {
        // sanity checking
    
        assert_eq!(
            random_normal([65536], ().seed(100)).reduce_std(()),
            tf32!(1.0004146)
        );

        assert_eq!(
            random_normal([65536], ().seed(10)).reduce_std(()),
            tf32!(1.0015763)
        );

        assert_eq!(
            random_normal([65536], ().seed(10).mean(10.)).reduce_std(()),
            tf32!(1.0015757)
        );

        assert_eq!(
            random_normal([65536], ().seed(10).stddev(10.)).reduce_std(()),
            tf32!(10.01571)
        );
    }

}
