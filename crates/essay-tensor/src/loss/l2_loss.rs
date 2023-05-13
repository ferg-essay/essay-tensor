use crate::{Tensor, ops::{fold_op, Fold}};

#[derive(Debug, Copy, Clone)]
pub struct L2Loss(f32);

pub fn l2_loss(a: &Tensor) -> Tensor {
    let n = a.dim_tail();
    let n_inv = 0.5 / n as f32;
    fold_op(a, 0.0.into(), L2Loss(n_inv))
}

impl Tensor {
    pub fn l2_loss(&self) -> Tensor {
        l2_loss(self)
    }
}

impl Fold for L2Loss {
    fn f(&self, acc: f32, a: f32) -> f32 {
        acc + self.0 * a * a
    }

    fn df_dx(&self, a: f32) -> f32 {
        a
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn l2_loss() {
        assert_eq!(tensor!(0.).l2_loss(), tensor!(0.));
        assert_eq!(tensor!(2.).l2_loss(), tensor!(2.));
        assert_eq!(tensor!(3.).l2_loss(), tensor!(4.5));
    }
}
