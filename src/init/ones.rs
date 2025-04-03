use crate::{Tensor, prelude::Shape};

/*
pub fn ones(shape: impl Into<Shape>) -> Tensor {
    init_op(Ones, shape)
}
    */

impl Tensor {
    /*
    pub fn ones(shape: impl Into<Shape>) -> Tensor {
        ones(shape)
    }
    */
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ones;
/*
impl InitKernel<f32> for Ones {
    type State = ();

    fn init(&self, _shape: &Shape) -> Self::State {
        ()
    }

    fn f(&self, _state: &mut Self::State) -> f32 {
        1.
    }
}
    */
/*
impl Initializer for Ones {
    fn init(&self, shape: &Shape) -> Tensor {
        init_op(self.clone(), shape)
    }
}
    */

#[cfg(test)]
mod test {
    use crate::prelude::*;
    use crate::init::ones;

    #[test]
    fn test_ones() {
        assert_eq!(ones([1]), tf32!([1.]));
        assert_eq!(ones([3]), tf32!([1., 1., 1.]));
        assert_eq!(ones([2, 3]), tf32!([[1., 1., 1.], [1., 1., 1.]]));
    }
}
