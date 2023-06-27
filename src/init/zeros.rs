use crate::{Tensor, prelude::Shape};
use crate::ops::{init_op, InitKernel};

use super::initializer::Initializer;

pub fn zeros(shape: impl Into<Shape>) -> Tensor {
    init_op(Zeros, shape)
}

pub fn zeros_initializer() -> Box<dyn Initializer> {
    Box::new(Zeros)
}

impl Tensor {
    pub fn zeros(shape: impl Into<Shape>) -> Tensor {
        zeros(shape)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Zeros;

impl InitKernel<f32> for Zeros {
    type State = ();

    fn init(&self, _shape: &Shape) -> Self::State {
        ()
    }

    fn f(&self, _state: &mut Self::State) -> f32 {
        0.
    }
}

impl Initializer for Zeros {
    fn init(&self, shape: &Shape) -> Tensor {
        init_op(self.clone(), shape)
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;
    use crate::init::zeros;

    #[test]
    fn test_zeros() {
        assert_eq!(zeros([1]), tf32!([0.]));
        assert_eq!(zeros([3]), tf32!([0., 0., 0.]));
        assert_eq!(zeros([2, 3]), tf32!([[0., 0., 0.], [0., 0., 0.]]));
    }
}