// Locator(TickHelper)

use essay_tensor::{Tensor, init::linspace};

pub trait Locator  {
    fn tick_values(&self, min: f32, max: f32) -> Tensor<f32>;

    fn view_limits(&self, min: f32, max: f32) -> (f32, f32) {
        (min, max)
    }
}

pub struct IndexLocator {
    base: f32,
    offset: f32,
}

impl IndexLocator {
    const MAXTICKS : usize = 1000;

    pub fn new(base: f32, offset: f32) -> Self {
        Self {
            base,
            offset,
        }
    }
}

impl Locator for IndexLocator {
    fn tick_values(&self, min: f32, max: f32) -> Tensor<f32> {
        let range = Tensor::arange(min + self.offset, max + 1., self.base);

        assert!(range.len() < Self::MAXTICKS);

        range
    }
}

pub struct LinearLocator {
    n_ticks: usize,
}

impl LinearLocator {
    const MAXTICKS : usize = 1000;

    pub fn new(n_ticks: Option<usize>) -> Self {
        let n_ticks = match n_ticks {
            Some(n_ticks) => n_ticks,
            None => 11
        };

        Self {
            n_ticks
        }
    }
}

impl Locator for LinearLocator {
    fn tick_values(&self, min: f32, max: f32) -> Tensor<f32> {
        let range = linspace(min, max, self.n_ticks);

        assert!(range.len() < Self::MAXTICKS);

        range
    }

    fn view_limits(&self, min: f32, max: f32) -> (f32, f32) {
        let (min, max) = if min < max {
            (min, max)
        } else if min == max {
            (min - 1., max + 1.)
        } else {
            (max, min)
        };

        let v_log10 = (max - min).log10();
        let n_log10 = ((self.n_ticks - 1).max(1) as f32).log10();
        let mut exp = v_log10 / n_log10;
        let rem = v_log10 % n_log10;

        if rem < 0.5 {
            exp -= 1.;
        }
        let scale = ((self.n_ticks - 1).max(1) as f32).powf(- exp);

        ((scale * min).floor() / scale, (scale * max).ceil() / scale)
    }
}
