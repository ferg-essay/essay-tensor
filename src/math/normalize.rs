use num_traits::Float;

use crate::tensor::{Tensor, Type};

impl<T: Type + Float + Clone> Tensor<T> {
    pub fn normalize_unit(&self) -> Self {
        self.normalize(None, 
            NormalizeUnit::default(),
            |s, v| s.update(v.clone()),
            |s| s,
            |s, v| s.norm(v.clone())
        )
    }
}

#[derive(Clone, Debug)]
struct NormalizeUnit<T: Float> {
    min: T,
    max: T,
}

impl<T: Float> NormalizeUnit<T> {
    fn update(self, v: T) -> Self {
        Self {
            min: self.min.min(v),
            max: self.max.max(v),
        }
    }

    fn norm(&self, v: T) -> T {
        (v - self.min) / (self.max - self.min)
    }
}

impl<T: Float> Default for NormalizeUnit<T> {
    fn default() -> Self {
        Self { 
            min: T::max_value(),
            max: T::min_value(),
        }
    }
}