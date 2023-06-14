// Locator(TickHelper)

use essay_tensor::{Tensor, init::linspace, tf32};

pub trait TickLocator  {
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

impl TickLocator for IndexLocator {
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

    pub fn _new(n_ticks: Option<usize>) -> Self {
        let n_ticks = match n_ticks {
            Some(n_ticks) => n_ticks,
            None => 11
        };

        Self {
            n_ticks
        }
    }
}

impl TickLocator for LinearLocator {
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
        /*
        let v_log10 = (max - min).log10();
        let n_log10 = ((self.n_ticks - 1).max(1) as f32).log10();
        let mut exp = v_log10 / n_log10;
        let rem = v_log10 % n_log10;

        if rem < 0.5 {
            exp -= 1.;
        }
        let scale = ((self.n_ticks - 1).max(1) as f32).powf(- exp);

        ((scale * min).floor() / scale, (scale * max).ceil() / scale)
        */
        (min, max)
    }
}

pub struct MaxNLocator {
    n_bins: usize,
    min_ticks: usize,
    steps: Tensor,
}

impl MaxNLocator {
    pub fn new(n_bins: Option<usize>) -> Self {
        let n_bins = match n_bins {
            Some(n_bins) => n_bins,
            None => 9
        };
        /*
        let steps = vec![
            1., 1.5, 2., 2.5, 3., 4., 5., 6., 8., 10.
        ];

        let steps = Tensor::from(steps);
        */
        let steps = vec![
            1., 2., 2.5, 5., 10.
        ];

        let steps = Tensor::from(steps);

        // TODO staircase

        Self {
            n_bins,
            min_ticks: 2,
            steps,
        }
    }

    fn raw_ticks(&self, vmin: f32, vmax: f32) -> Tensor<f32> {
        let n_bins = self.n_bins;
        // let (vmin, vmax) = (0., 1.);
        let (scale, offset) = scale_range(vmin, vmax, n_bins);

        let vmin = vmin - offset;
        let vmax = vmax - offset;
        let raw_step = (vmax - vmin) / n_bins as f32;
        let steps = &self.steps * scale;
        let istep = steps.iter().position(|s| raw_step < *s ).unwrap();

        let mut ticks: Tensor = tf32!([1.]);
        
        for i in (0..=istep).rev() {
            let step = steps[i];
            let best_vmin = (vmin / step).trunc() * step;
            // TODO: see matplotlib handling of floating point limits
            //let low = (vmin - best_vmin) / step;
            let low = best_vmin / step;
            let high = (vmax - vmin + best_vmin) / step;
            ticks = Tensor::arange(low, high + 1., 1.) * step + best_vmin;

            let n_ticks = ticks.iter()
                .filter(|s| vmin <= **s && **s <= vmax)
                .count();

            if self.min_ticks <= n_ticks && n_ticks < n_bins {
                break;
            }
        }

        return ticks + offset;
    }
}

impl TickLocator for MaxNLocator {
    fn tick_values(&self, vmin: f32, vmax: f32) -> Tensor<f32> {
        let (vmin, vmax) = nonsingular(vmin, vmax, 1e-13, 1e-14);

        let ticks = self.raw_ticks(vmin, vmax);

        ticks
    }

    fn view_limits(&self, min: f32, max: f32) -> (f32, f32) {
        let (min, max) = if min < max {
            (min, max)
        } else {
            (max, min)
        };

        nonsingular(min, max, 1e-12, 1e-13)
    }
}

fn nonsingular(min: f32, max: f32, expander: f32, tiny: f32) -> (f32, f32) {
    if tiny < max - min {
        (min, max)
    }  else {
        (min, min + expander)
    }
}

fn scale_range(vmin: f32, vmax: f32, n_bins: usize) -> (f32, f32) {
    let threshold = 100.;

    let dv = (vmax - vmin).abs();
    let vmid = (vmin + vmax) / 2.;

    let offset = if vmid.abs() / dv < threshold {
        0.
    } else {
        10.0f32.powf(vmid.abs().log10().floor()).copysign(vmid)
    };

    let scale = 10.0f32.powf((dv / n_bins as f32).log10().floor());

    (scale, offset)
}