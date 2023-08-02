use crate::Tensor;

pub fn histogram(data: impl Into<Tensor>, args: impl Into<HistArgs>) -> Tensor {
    let data : Tensor = data.into();
    assert!(data.rank() == 1, "histogram requires a rank 1 tensor {:?}", data.shape().as_slice());

    let args = args.into();
    let min = data.reduce_min()[0];
    let max = data.reduce_max()[0];
    let delta = if min == max { 
        f32::EPSILON
    } else { 
        (max - min) / args.n_bins as f32 
    };

    let mut values = Vec::<f32>::new();
    values.resize(args.n_bins, 0.);

    for item in data.iter() {
        let bin = ((*item - min) / delta).floor() as usize;

        if bin < args.n_bins {
            values[bin] += 1.;
        } else {
            values[args.n_bins - 1] += 1.;
        }
    }

    Tensor::from(values)
}

pub struct HistArgs {
    n_bins: usize,
}

impl HistArgs {
    pub fn new() -> Self {
        HistArgs::default()
    }

    pub fn n_bins(mut self, n_bins: usize) -> Self {
        assert!(n_bins > 0);

        self.n_bins = n_bins;

        self
    }
}

impl Default for HistArgs {
    fn default() -> Self {
        Self { 
            n_bins: 10
        }
    }
}

impl From<()> for HistArgs {
    fn from(_: ()) -> Self {
        HistArgs::default()
    }
}

impl From<usize> for HistArgs {
    fn from(n_bins: usize) -> Self {
        assert!(n_bins > 0);

        HistArgs::default().n_bins(n_bins)
    }
}

#[cfg(test)]
mod test {
    use crate::{prelude::*, stats::histogram};

    #[test]
    fn histogram_default() {
        assert_eq!(
            histogram(tf32!([0., 1., 2., 1., 3., 9.]), ()),
            tf32!([1., 2., 1., 1., 0., 0., 0., 0., 0., 1.])
        );

        assert_eq!(
            histogram(tf32!([0., 10., 1., 1.9, 1.5]), ()),
            tf32!([1., 3., 0., 0., 0., 0., 0., 0., 0., 1.])
        );
    }

    #[test]
    fn histogram_bins() {
        assert_eq!(
            histogram(tf32!([0., 1., 2., 1., 3.]), 4),
            tf32!([1., 2., 1., 1.])
        );
    }
}