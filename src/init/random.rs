use crate::random::Rand32;
use crate::tensor::{Tensor, Shape};

pub fn random_normal(
    shape: impl Into<Shape>,
    opt: impl Into<Normal>,
) -> Tensor {
    let opt = opt.into();

    let mut rand = opt.init();
    Tensor::init(shape, move || {
        let p = rand.next_normal();

        opt.mean + p * opt.std
    })
}

pub fn random_uniform(
    shape: impl Into<Shape>,
    opt: impl Into<Uniform>,
) -> Tensor {
    let opt = opt.into();

    let mut rand = opt.init();
    Tensor::init(shape, move || {
        let p = rand.next_normal();

        opt.min + p * (opt.max - opt.min).max(0.)
    })
}

impl Tensor {
    pub fn random_normal(shape: impl Into<Shape>, opt: impl Into<Normal>) -> Tensor {
        random_normal(shape, opt)
    }

    pub fn random_uniform(shape: impl Into<Shape>, opt: impl Into<Uniform>) -> Tensor {
        random_uniform(shape, opt)
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Normal {
    mean: f32,
    std: f32,
    seed: Option<u64>,
}

impl Normal {
    pub fn mean(mean: f32) -> Self {
        Self {
            mean,
            std: 1.,
            seed: None,
        }
    }

    pub fn with_mean(self, mean: f32) -> Self {
        Self {
            mean,
            ..self
        }
    }

    pub fn std(std: f32) -> Self {
        Self {
            mean: 0.,
            std,
            seed: None,
        }
    }

    pub fn with_std(self, std: f32) -> Self {
        Self {
            std,
            ..self
        }
    }

    pub fn seed(seed: u64) -> Self {
        Self {
            mean: 0.,
            std: 1.,
            seed: Some(seed),
        }
    }

    pub fn with_seed(self, seed: u64) -> Self {
        Self {
            seed: Some(seed),
            ..self
        }
    }

    fn init(&self) -> Rand32 {
        match self.seed {
            Some(seed) => Rand32(seed),
            None => Rand32::new(),
        }
    }
}

impl Default for Normal {
    fn default() -> Self {
        Self { 
            mean: 0.,
            std: 1.,
            seed: None,
        }
    }
}

impl From<Option<u64>> for Normal {
    fn from(seed: Option<u64>) -> Self {
        Self {
            mean: 0.,
            std: 1.,
            seed,
        }
    }
}

impl From<(f32, f32)> for Normal {
    fn from(mean_std: (f32, f32)) -> Self {
        Self::mean(mean_std.0).with_std(mean_std.1)
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Uniform {
    min: f32,
    max: f32,
    seed: Option<u64>,
}

impl Uniform {
    pub fn min(min: f32) -> Self {
        Self {
            min,
            max: 1.,
            seed: None,
        }
    }

    pub fn with_min(self, min: f32) -> Self {
        Self {
            min,
            ..self
        }
    }

    pub fn max(max: f32) -> Self {
        Self {
            min: 0.,
            max,
            seed: None,
        }
    }

    pub fn with_max(self, max: f32) -> Self {
        Self {
            max,
            ..self
        }
    }

    pub fn seed(seed: u64) -> Self {
        Self {
            min: 0.,
            max: 1.,
            seed: Some(seed),
        }
    }

    pub fn with_seed(self, seed: u64) -> Self {
        Self {
            seed: Some(seed),
            ..self
        }
    }

    fn init(&self) -> Rand32 {
        match self.seed {
            Some(seed) => Rand32(seed),
            None => Rand32::new(),
        }
    }
}

impl Default for Uniform {
    fn default() -> Self {
        Self { 
            min: 0.,
            max: 1.,
            seed: None,
        }
    }
}

impl From<Option<u64>> for Uniform {
    fn from(seed: Option<u64>) -> Self {
        Self {
            min: 0.,
            max: 1.,
            seed,
        }
    }
}

impl From<(f32, f32)> for Uniform {
    fn from(min_max: (f32, f32)) -> Self {
        Self::min(min_max.0).with_max(min_max.1)
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;
    use crate::init::{random_normal, random_uniform, Normal, Uniform};

    #[test]
    fn normal_seed() {
        assert_eq!(
            random_normal([1], Some(100)), 
            ten![0.28612638]
        );

        // check seed is consistent
        assert_eq!(
            random_normal([1], Normal::seed(100)), 
            ten![0.28612638]
        );
        
        assert_eq!(
            random_normal([4], Normal::seed(100)), 
            ten![0.28612638, -0.4481697, -1.3627453, 0.61000276]
        );
        
        assert_eq!(
            random_normal([2, 2], Normal::seed(100)), 
            ten![[0.28612638, -0.4481697], [-1.3627453, 0.61000276]]
        );
        
        assert_eq!(
            random_normal([3, 2, 2], Normal::seed(100)), 
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
            random_normal([4], Normal::seed(100)), 
            ten![0.28612638, -0.4481697, -1.3627453, 0.61000276]
        );

        assert_eq!(
            random_normal([4], Normal::seed(100).with_mean(100.)), 
            ten![100.286125, 99.551834, 98.63725, 100.61]
        );

        assert_eq!(
            random_normal([4], Normal::seed(100).with_mean(100.)), 
            ten![100.286125, 99.551834, 98.63725, 100.61]
        );

        assert_eq!(
            random_normal([4], Normal::seed(100).with_mean(-10.)), 
            ten![-9.713874, -10.44817, -11.362745, -9.3899975]
        );
    }

    #[test]
    fn test_normal_stddev() {
        // base normal distribution
        assert_eq!(
            random_normal([4], Some(100)), 
            ten![0.28612638, -0.4481697, -1.3627453, 0.61000276]
        );

        assert_eq!(
            random_normal([4], Normal::std(10.).with_seed(100)),
            ten![2.8612638, -4.481697, -13.627453, 6.1000276]
        );

        assert_eq!(
            random_normal([4], Normal::seed(100).with_std(100.)), 
            ten![28.612637, -44.81697, -136.27454, 61.000275]
        );
    }

    #[test]
    fn test_normal_reduce_mean() {
        // sanity checking

        assert_eq!(
            random_normal([65536], Some(100)).reduce_mean(),
            Tensor::from(5.9156395e-5)
        );

        assert_eq!(
            random_normal([65536], Some(10)).reduce_mean(),
            Tensor::from(-0.0028584553)
        );

        assert_eq!(
            random_normal([65536], Normal::seed(100).with_mean(2.)).reduce_mean(),
            Tensor::from(2.0000522)
        );

        assert_eq!(
            random_normal([65536], Normal::seed(10).with_std(10.)).reduce_mean(),
            Tensor::from(-0.028584845)
        );
    }


    #[test]
    fn test_normal_reduce_std() {
        // sanity checking
    
        assert_eq!(
            random_normal([65536], Some(100)).reduce_std(),
            Tensor::from(0.99907887)
        );

        assert_eq!(
            random_normal([65536], Some(10)).reduce_std(),
            Tensor::from(0.9967639)
        );

        assert_eq!(
            random_normal([65536], Normal::seed(10).with_mean(10.)).reduce_std(),
            Tensor::from(0.9967649)
        );

        assert_eq!(
            random_normal([65536], Normal::seed(10).with_std(10.)).reduce_std(),
            Tensor::from(9.967605)
        );
    }

    #[test]
    fn test_uniform() {
        assert_eq!(
            random_uniform([1], Some(100)), 
            ten![0.28612638]
        );

        // check seed is consistent
        assert_eq!(
            random_uniform([1], Uniform::seed(100)), 
            ten![0.28612638]
        );
        
        assert_eq!(
            random_uniform([4], Uniform::seed(100)), 
            ten![0.28612638, -0.4481697, -1.3627453, 0.61000276]
        );
        
        assert_eq!(
            random_uniform([2, 2], Some(100)), 
            ten![[0.28612638, -0.4481697], [-1.3627453, 0.61000276]]
        );
        
        assert_eq!(
            random_uniform([3, 2, 2], Some(100)), 
            ten![
                [[0.28612638, -0.4481697],
                 [-1.3627453, 0.61000276]],

                [[0.5581478, 1.159134],
                [-0.5566373, -0.2761264]],

                [[0.9464901, -0.254632],
                [1.2101488, -0.34107557]]
            ]
        );
    }
}
