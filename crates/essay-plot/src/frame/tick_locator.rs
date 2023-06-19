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
        // let steps = vec![1., 1.5, 2., 2.5, 3., 4., 5., 6., 8., 10.];

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

    pub fn steps(&mut self, steps: &[f32]) -> &mut Self {
        assert!(steps.len() > 0);

        let mut vec = Vec::<f32>::new();

        if steps[0] != 1. {
            vec.push(steps[0]);
        }

        for step in steps {
            assert!(1.0 <= *step && *step <= 10.);

            vec.push(*step);
        }

        if steps[steps.len() - 1] != 10. {
            vec.push(10.);
        }

        self.steps = Tensor::from(steps);

        self
    }

    fn raw_ticks(&self, vmin: f32, vmax: f32) -> Tensor<f32> {
        let n_bins = self.n_bins;
        // let (vmin, vmax) = (0., 1.);
        let (scale, offset) = scale_range(vmin, vmax, n_bins);

        let vmin = vmin - offset;
        let vmax = vmax - offset;
        let raw_step = (vmax - vmin) / n_bins as f32;
        let steps = &self.steps * scale;
        let istep = match steps.iter().position(|s| raw_step < *s ) {
            Some(istep) => istep,
            None => 0,
        };

        let mut ticks: Tensor = tf32!([1.]);
        
        for i in (0..=istep).rev() {
            let step = steps[i];
            let best_vmin = (vmin / step).trunc() * step;

            let low = best_min(vmin - best_vmin, step, offset);
            let high = best_max(vmax - best_vmin, step, offset);
            ticks = Tensor::arange(low, high + 1., 1.) * step + best_vmin;

            let n_ticks = ticks.iter()
                .filter(|s| vmin <= **s && **s <= vmax)
                .count();

            if self.min_ticks <= n_ticks {
                break;
            }
        }

        return ticks + offset;
    }
}

fn best_min(vmin_offset: f32, step: f32, _offset: f32) -> f32 {
    let low = (vmin_offset / step).round();
    low
}

fn best_max(vmin_offset: f32, step: f32, _offset: f32) -> f32 {
    let high = (vmin_offset / step).round();
    high
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
        } else if min == max {
            (min - 1., max + 1.)
        } else {
            (max, min)
        };

        let (min, max) = nonsingular(min, max, 1e-12, 1e-13);

        let is_round_numbers = true;

        if is_round_numbers {
            let ticks = self.raw_ticks(min, max);

            (ticks[0], ticks[ticks.len() - 1])
        } else {
            (min, max)
        }
    }
}

fn nonsingular(min: f32, max: f32, expander: f32, tiny: f32) -> (f32, f32) {
    if ! min.is_finite() || ! max.is_finite() {
        return (-expander, expander);
    }

    if tiny < max - min {
        (min, max)
    } else if min == 0. && max == 0. {
        (-expander, expander)
    } else {
        (min - min.abs() * expander, max + max.abs() * expander)
    }
}

fn scale_range(vmin: f32, vmax: f32, n_bins: usize) -> (f32, f32) {
    let threshold = 100.;

    let dv = (vmax - vmin).abs();

    if dv == 0. {
        return (1., 0.);
    }

    let vmid = (vmin + vmax) / 2.;

    let offset = if vmid.abs() / dv < threshold {
        0.
    } else {
        10.0f32.powf(vmid.abs().log10().floor()).copysign(vmid)
    };

    let scale = 10.0f32.powf((dv / n_bins as f32).log10().floor());

    (scale, offset)
}

#[cfg(test)]
mod test {
    use essay_tensor::tf32;

    use crate::frame::tick_locator::TickLocator;

    use super::MaxNLocator;

    #[test]
    fn max_n_locator_view_limits() {
        let mut locator = MaxNLocator::new(Some(9));;
        locator.steps(&vec![1., 2., 2.5, 5., 10.]);

        assert_eq!(locator.view_limits(0., 0.), (-1., 1.));
        assert_eq!(locator.view_limits(1., 1.), (0., 2.));

        assert_eq!(locator.view_limits(1., 0.), (0., 1.));
        assert_eq!(locator.view_limits(-1., 0.), (-1., 0.));
        assert_eq!(locator.view_limits(0., -1.), (-1., 0.));

        assert_eq!(locator.view_limits(0., 1.), (0., 1.));
        assert_eq!(locator.view_limits(0., 1.0e-6), (0., 1.0e-6));

        assert_eq!(locator.view_limits(1., 1. + 1.0e-6), (1., 1.000001));
    }

    #[test]
    fn max_n_locator_tick_values_0_1() {
        let mut locator = MaxNLocator::new(Some(9));;
        locator.steps(&vec![1., 2., 2.5, 5., 10.]);

        assert_eq!(
            locator.tick_values(0., 1.0), 
            tf32!([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        );

        assert_eq!(
            locator.tick_values(0., 10.0), 
            tf32!([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
        );

        assert_eq!(
            locator.tick_values(0., 0.1), 
            tf32!([0.0, 0.02, 0.04, 0.06, 0.08, 0.099999994])
        );

        assert_eq!(
            locator.tick_values(0., 1e6), 
            tf32!([0.0, 2e5, 4e5, 6e5, 8e5, 10e5])
        );
    }

    #[test]
    fn max_n_locator_tick_values_offset() {
        let mut locator = MaxNLocator::new(Some(9));;
        locator.steps(&vec![1., 2., 2.5, 5., 10.]);

        assert_eq!(
            locator.tick_values(11., 12.0), 
            tf32!([11.0, 11.2, 11.4, 11.6, 11.8, 12.0])
        );

        assert_eq!(
            locator.tick_values(-1009., -1008.0), 
            tf32!([-1009.0, -1008.8, -1008.6, -1008.4, -1008.2, -1008.0])
        );
    }

    #[test]
    fn max_n_locator_tick_values_ranges() {
        let mut locator = MaxNLocator::new(Some(9));;
        locator.steps(&vec![1., 2., 2.5, 5., 10.]);

        assert_eq!(
            locator.tick_values(10., 11.0), 
            tf32!([10.0, 10.2, 10.4, 10.6, 10.8, 11.0])
        );

        assert_eq!(
            locator.tick_values(10., 12.0), 
            tf32!([10.0, 10.25, 10.5, 10.75, 11., 11.25, 11.5, 11.75, 12.])
        );

        assert_eq!(
            locator.tick_values(10., 13.0), 
            tf32!([10.0, 10.5, 11.0, 11.5, 12., 12.5, 13.0])
        );

        assert_eq!(
            locator.tick_values(10., 14.0), 
            tf32!([10.0, 10.5, 11.0, 11.5, 12., 12.5, 13.0, 13.5, 14.])
        );

        assert_eq!(
            locator.tick_values(10., 15.0), 
            tf32!([10.0, 11., 12., 13., 14., 15.])
        );

        assert_eq!(
            locator.tick_values(10., 16.0), 
            tf32!([10.0, 11., 12., 13., 14., 15., 16.])
        );

        assert_eq!(
            locator.tick_values(10., 17.0), 
            tf32!([10.0, 11., 12., 13., 14., 15., 16., 17.])
        );

        assert_eq!(
            locator.tick_values(10., 18.0), 
            tf32!([10.0, 11., 12., 13., 14., 15., 16., 17., 18.])
        );

        assert_eq!(
            locator.tick_values(10., 19.0), 
            tf32!([10.0, 12., 14., 16., 18., 20.])
        );

        assert_eq!(
            locator.tick_values(10., 20.0), 
            tf32!([10.0, 12., 14., 16., 18., 20.])
        );
    }

    #[test]
    fn max_n_locator_tick_zero() {
        let mut locator = MaxNLocator::new(Some(9));;
        locator.steps(&vec![1., 2., 2.5, 5., 10.]);

        assert_eq!(
            locator.tick_values(-2., 6.), 
            tf32!([-2., -1., 0., 1., 2., 3., 4., 5., 6.])
        );

        assert_eq!(
            locator.tick_values(-2.4, 6.28), 
            tf32!([-2., -1., 0., 1., 2., 3., 4., 5., 6.])
        );
    }
}