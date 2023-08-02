use crate::{Tensor, init::linspace};

pub fn histogram2d(data: impl Into<Tensor>, args: impl Into<Hist2Args>) -> (Tensor, Tensor, Tensor) {
    let data : Tensor = data.into();
    assert!(data.rank() == 2, "histogram2d requires a rank 2 tensor {:?}", data.shape().as_slice());
    assert!(data.cols() == 2, "histogram2d requires 2D tensor {:?}", data.shape().as_slice());

    let args = args.into();
    let min = data.reduce_min_opt(0);
    let max = data.reduce_max_opt(0);
    let (min_x, min_y) = (min[0], min[1]);
    let (max_x, max_y) = (max[0], max[1]);

    let delta_x = if min_x == max_x { 
        f32::EPSILON
    } else { 
        (max_x - min_x) / args.n_bins as f32 
    };

    let delta_y = if min_y == max_y { 
        f32::EPSILON
    } else { 
        (max_y - min_y) / args.n_bins as f32 
    };

    let bins_x = linspace(min_x, max_x, args.n_bins + 1);
    let bins_y = linspace(min_y, max_y, args.n_bins + 1);

    let mut values = Vec::<f32>::new();
    values.resize(args.n_bins * args.n_bins, 0.);

    for item in data.iter_row() {
        let bin_x = (((item[0] - min_x) / delta_x).floor() as usize).min(args.n_bins - 1);
        let bin_y = (((item[1] - min_y) / delta_y).floor() as usize).min(args.n_bins - 1);

        values[bin_y * args.n_bins + bin_x] += 1.;
    }

    (
        Tensor::from(values).reshape([args.n_bins, args.n_bins]),
        bins_x,
        bins_y
    )
}

pub struct Hist2Args {
    n_bins: usize,
}

impl Hist2Args {
    pub fn new() -> Self {
        Hist2Args::default()
    }

    pub fn n_bins(mut self, n_bins: usize) -> Self {
        assert!(n_bins > 0);

        self.n_bins = n_bins;

        self
    }
}

impl Default for Hist2Args {
    fn default() -> Self {
        Self { 
            n_bins: 10
        }
    }
}

impl From<()> for Hist2Args {
    fn from(_: ()) -> Self {
        Hist2Args::default()
    }
}

impl From<usize> for Hist2Args {
    fn from(n_bins: usize) -> Self {
        assert!(n_bins > 0);

        Hist2Args::default().n_bins(n_bins)
    }
}

#[cfg(test)]
mod test {
    use crate::{prelude::*, stats::histogram2d};

    #[test]
    fn histogram2d_bins_2() {
        assert_eq!(
            histogram2d(tf32!([[0., 0.], [1., 0.], [1., 1.], [0.51, 0.49]]), 2), (
                tf32!([[1., 2.], [0., 1.]]),
                tf32!([0., 0.5, 1.]),
                tf32!([0., 0.5, 1.]),
            )
        );
    }
}