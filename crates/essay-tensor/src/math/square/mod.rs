use crate::{ops::Uop};

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct SquareOp;

impl Uop<f32> for SquareOp {
    fn f(&self, value: f32) -> f32 {
        value * value
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn square() {
        assert_eq!(tensor!(2.).square(), tensor!(4.));
        assert_eq!(tensor!([3., 4.]).square(), tensor!([9., 16.]));
    }
}
