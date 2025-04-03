use std::{cmp, f32::consts::PI};

use essay_opt::derive_opt;
use rustfft::{FftPlanner, num_complex::Complex};

use crate::{Tensor, tensor::TensorUninit};

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
        let data = TensorUninit::<f32>::create(batch * len_out, |o| {
            for n in 0..batch {
                let x = tensor.as_wrap_slice(n * len);

                for i in 0..len {
                    buffer[i] = Complex { re: x[i] * window[i], im: 0. };
                }
        
                fft_fwd.process(&mut buffer);

                let fft_slice = buffer.as_slice();
                let offset = n * len_out;

                for i in 0..cmp::min(fft_out, len_out) {
                    o[offset + i] = fft_slice[i].norm();
                }

                for i in fft_out..len_out {
                    o[offset + i] = 0.;
                }
            }
        });


        // let mut vec = Vec::from(tensor.shape().as_slice());
        // let len = vec.len();
        // vec[len - 1] = len_out;
        data.into_tensor(tensor.shape().with_col(len_out))
    }
}

fn hann_window(len: usize) -> Tensor {
    unsafe {
        TensorUninit::<f32>::create(len, |o| {
            let step : f32 = PI / len as f32;

            for i in 0..len {
                let tmp = (step * i as f32).sin();

                o[i] = tmp * tmp;
            }
        }).into_tensor([len])
    }
}

#[derive_opt(FftOpt)]
#[derive(Default)]
pub struct FftArg {
    fft_length: Option<usize>,
}

#[cfg(test)]
mod test {
    use crate::{prelude::*, signal::rfft_norm};
    use super::FftOpt;

    #[test]
    fn test_fft_norm() {
        assert_eq!(
            rfft_norm(tf32!([0., 0., 1., 1., 0., 0., 1., 1.]), ()),
            tf32!([4.0, 0., 2.828427, 0., 0.])
        );

        assert_eq!(
            rfft_norm(tf32!([0., 0., 1., 1.]), ()),
            tf32!([2.0, 1.4142135, 0.0])
        );

        assert_eq!(
            rfft_norm(tf32!([0., 0., 1., 1., 0., 0., -1., -1.]), ()),
            tf32!([0.0, 3.6955183, 0.0, 1.5307337, 0.])
        );

        assert_eq!(
            rfft_norm(tf32!([0., 1., 0., 1., 0., 1., 0., 1.]), ()),
            tf32!([4.0, 0., 0., 0., 4.])
        );
    }

    #[test]
    fn test_fft_shape() {
        assert_eq!(
            rfft_norm(tf32!([[0., 0., 1., 1.], [0., 0., 1., 1.]]), ()),
            tf32!([[2.0, 1.4142135, 0.], [2.0, 1.4142135, 0.]]),
        );

        assert_eq!(
            rfft_norm(tf32!([[[0., 0., 1., 1.], [0., 0., 1., 1.]]]), ()),
            tf32!([[[2.0, 1.4142135, 0.], [2.0, 1.4142135, 0.]]]),
        );
    }

    #[test]
    fn test_fft_len() {
        assert_eq!(
            rfft_norm(tf32!([[0., 0., 1., 1.], [0., 0., 1., 1.]]), ().fft_length(4)),
            tf32!([[2.0, 1.4142135, 0., 0.], [2.0, 1.4142135, 0., 0.]]),
        );

        assert_eq!(
            rfft_norm(tf32!([[[0., 0., 1., 1.], [0., 0., 1., 1.]]]), ().fft_length(2)),
            tf32!([[[2.0, 1.4142135], [2.0, 1.4142135]]]),
        );
    }
}