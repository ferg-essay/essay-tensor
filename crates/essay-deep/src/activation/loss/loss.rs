use crate::Tensor;

pub trait Loss {
    fn loss(&self, y_true: &Tensor, y_pred: &Tensor) -> Tensor;
}

impl<F> Loss for F 
where
    F: Fn(&Tensor, &Tensor) -> Tensor,
{
    fn loss(&self, y_true: &Tensor, y_pred: &Tensor) -> Tensor {
        self(y_true, y_pred)
    }
}