use std::{cmp, f32::consts::PI};

use essay_opt::derive_opt;
use rustfft::{FftPlanner, num_complex::Complex};

use crate::tensor::{Tensor, unsafe_init};

pub fn rfft_norm(tensor: impl Into<Tensor>, opt: impl FftOpt) -> Tensor {
    let tensor = tensor.into();
    let opt = opt.into_arg();
    let len = tensor.cols();
    let batch = tensor.len() / len;

    let mut planner = FftPlanner::<f32>::new();
    let fft_fwd = planner.plan_fft_forward(len);

    let mut buffer = Vec::<Complex<f32>>::new();
    buffer.resize(len, Complex { re: 0., im: 0. });

    let window = hann_window(len);

    let fft_out = (len / 2) + 1;
    let len_out = match opt.fft_length {
        Some(fft_length) => fft_length,
        None => fft_out,
    };

    unsafe {
        let shape = tensor.shape().clone().with_cols(len_out);

        unsafe_init::<f32>(batch * len_out, shape, |o| {
            for n in 0..batch {
                let x = tensor.as_wrap_slice(n * len);

                for i in 0..len {
                    buffer[i] = Complex { re: x[i] * window[i], im: 0. };
                }
        
                fft_fwd.process(&mut buffer);

                let fft_slice = buffer.as_slice();
                let offset = n * len_out;

                for i in 0..cmp::min(fft_out, len_out) {
                    o.add(offset + i).write(fft_slice[i].norm());
                }

                for i in fft_out..len_out {
                    o.add(offset + i).write(0.);
                }
            }
        })
    }
}

fn hann_window(len: usize) -> Tensor {
    let step : f32 = PI / len as f32;
    
    Tensor::init_rindexed([len], |idx| {
        let tmp = (step * idx[0] as f32).sin();
        tmp * tmp
    })
}

#[derive_opt(FftOpt)]
#[derive(Default)]
pub struct FftArg {
    fft_length: Option<usize>,
}

#[cfg(test)]
mod test {
    use crate::{signal::rfft_norm, ten};
    use super::FftOpt;

    #[test]
    fn test_fft_norm() {
        assert_eq!(
            rfft_norm(ten!([0., 0., 1., 1., 0., 0., 1., 1.]), ()),
            ten!([4.0, 0., 2.828427, 0., 0.])
        );

        assert_eq!(
            rfft_norm(ten!([0., 0., 1., 1.]), ()),
            ten!([2.0, 1.4142135, 0.0])
        );

        assert_eq!(
            rfft_norm(ten!([0., 0., 1., 1., 0., 0., -1., -1.]), ()),
            ten!([0.0, 3.6955183, 0.0, 1.5307337, 0.])
        );

        assert_eq!(
            rfft_norm(ten!([0., 1., 0., 1., 0., 1., 0., 1.]), ()),
            ten!([4.0, 0., 0., 0., 4.])
        );
    }

    #[test]
    fn test_fft_shape() {
        assert_eq!(
            rfft_norm(ten!([[0., 0., 1., 1.], [0., 0., 1., 1.]]), ()),
            ten!([[2.0, 1.4142135, 0.], [2.0, 1.4142135, 0.]]),
        );

        assert_eq!(
            rfft_norm(ten!([[[0., 0., 1., 1.], [0., 0., 1., 1.]]]), ()),
            ten!([[[2.0, 1.4142135, 0.], [2.0, 1.4142135, 0.]]]),
        );
    }

    #[test]
    fn test_fft_len() {
        assert_eq!(
            rfft_norm(ten!([[0., 0., 1., 1.], [0., 0., 1., 1.]]), ().fft_length(4)),
            ten!([[2.0, 1.4142135, 0., 0.], [2.0, 1.4142135, 0., 0.]]),
        );

        assert_eq!(
            rfft_norm(ten!([[[0., 0., 1., 1.], [0., 0., 1., 1.]]]), ().fft_length(2)),
            ten!([[[2.0, 1.4142135], [2.0, 1.4142135]]]),
        );
    }
}