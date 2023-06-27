use crate::{Tensor, ops::{NormalizeKernel, normalize_op}};

#[derive(Debug, Copy, Clone)]
pub struct NormalizeUnit;

pub fn normalize_unit(x: &Tensor) -> Tensor {
    normalize_op(x, NormalizeUnit, ())
}

impl Tensor {
    pub fn normalize_unit(&self) -> Tensor {
        normalize_unit(self)
    }
}

impl NormalizeKernel for NormalizeUnit {
    type State = NormalizeState;
    
    #[inline]
    fn init(&self) -> Self::State {
        Self::State {
            min: f32::MAX,
            max: f32::MIN,
        }
    }
    
    #[inline]
    fn accum(&self, state: Self::State, value: f32) -> Self::State {
        Self::State {
            min: state.min.min(value),
            max: state.max.max(value)
        }
    }
    
    #[inline]
    fn f(&self, state: &Self::State, v: f32) -> f32 {
        let norm = (v - state.min) / (state.max - state.min);

        norm
    }

    #[inline]
    fn df_dx(&self, _x: f32) -> f32 {
        todo!()
    }
}

pub struct NormalizeState {
    min: f32,
    max: f32
}
