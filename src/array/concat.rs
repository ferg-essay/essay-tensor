use crate::tensor::{Axis, IntoTensorList, Tensor, TensorData, Type};

pub fn concatenate<T>(x: impl IntoTensorList<T>) -> Tensor<T>
where
    T: Type + Clone
{
    let mut vec = Vec::<Tensor<T>>::new();

    x.into_list(&mut vec);

    concat_axis(vec, 0)
}

pub fn concatenate_axis<T>(axis: impl Into<Axis>, x: impl IntoTensorList<T>) -> Tensor<T>
where
    T: Type + Clone
{
    let mut vec = Vec::<Tensor<T>>::new();

    let axis: Axis = axis.into();

    x.into_list(&mut vec);

    // concat_vec(vec, axis)
    let axis = if vec.len() > 0 {
        axis.axis_from_rank(vec[0].rank())
    } else {
        0
    };

    concat_axis(vec, axis)
}

pub(crate) fn concat_axis<T: Type + Clone>(
    args: Vec<Tensor<T>>, 
    axis: usize,
) -> Tensor<T> {
    let shape = args[0].shape();

    let axis_len = validate_concat(&args, axis);

    let shape_inner = shape.clone().remove(axis);

    let o_len = args.iter().map(|t| t.size()).sum();
    let n_outer = shape_inner.sublen(0, axis);
    let n_inner = if axis < shape_inner.rank() {
        shape_inner.sublen(axis, shape_inner.rank())
    } else {
        1
    };

    let shape = shape_inner.insert(axis, axis_len);

    unsafe {
        TensorData::<T>::unsafe_init(o_len, |o| {
            let j_stride = n_inner * axis_len;
            let mut t = 0;

            for tensor in args.iter() {
                let n_axis = tensor.shape().dim(axis);

                let x = tensor.as_slice();

                for j in 0..n_outer {
                    for k in 0..n_axis {
                        for i in 0..n_inner {
                            let value = x[j * n_axis * n_inner + k * n_inner + i].clone();

                            o.add(j * j_stride + t + k * n_inner + i).write(value);
                        }
                    }
                }

                t += n_axis * n_inner;
            }

        }).into_tensor(shape)
    }
}

fn validate_concat<T: Type>(args: &Vec<Tensor<T>>, axis: usize) -> usize {
    let shape = args[0].shape();

    let mut axis_len = 0;

    // TODO: split out validation
    for x in args {
        let x_shape = x.shape();
    
        assert_eq!(shape.rank(), x_shape.rank(),
            "concat() tensor shape must match {:?} {:?}", 
            shape.as_vec(), x_shape.as_vec()
        );
    
        for i in 0..shape.rank() {
            if i == axis {
                axis_len += x_shape.dim(axis);
            } else {
                assert_eq!(x_shape.dim(i), shape.dim(i),
                    "concat() tensor shape must match {:?} {:?}", 
                    shape.as_vec(), x_shape.as_vec()
                );
            }
        }
    }

    axis_len
}

#[cfg(test)]
mod test {
    use crate::{array::{concatenate, concatenate_axis}, prelude::*};
    
    #[test]
    fn test_concat() {
        assert_eq!(concatenate((
            tf32!([1.]),
            tf32!([10.])
        )), tf32!([1., 10.]));

        assert_eq!(concatenate((
            tf32!([1., 2.]),
            tf32!([10., 20., 30.])
        )), tf32!([
            1., 2., 10., 20., 30.,
        ]));

        assert_eq!(concatenate((
            tf32!([[1.], [2.]]),
            tf32!([[10.], [20.], [30.]])
        )), tf32!([
            [1.], [2.], [10.], [20.], [30.],
        ]));

        assert_eq!(concatenate((
            tf32!([[1., 2.]]),
            tf32!([[10., 20.], [30., 40.]])
        )), tf32!([
            [1., 2.], [10., 20.], [30., 40.],
        ]));

        assert_eq!(concatenate((
            tf32!([[[1., 2.]]]),
            tf32!([[[10., 20.]], [[30., 40.]]])
        )), tf32!([
            [[1., 2.]], [[10., 20.]], [[30., 40.]],
        ]));
    }
    
    #[test]
    fn test_concat_axis() {
        assert_eq!(concatenate_axis(0, (
            tf32!([1.]),
            tf32!([10.])
        )), tf32!([
            1., 10., 
        ]));

        assert_eq!(concatenate_axis(-1, (
            tf32!([1., 2.]),
            tf32!([10., 20., 30.])
        )), tf32!([
            1., 2., 10., 20., 30.,
        ]));

        assert_eq!(concatenate_axis(Axis::axis(0), (
            tf32!([[1., 2.]]),
            tf32!([[10., 20.]])
        )), tf32!([
            [1., 2.], [10., 20.],
        ]));

        assert_eq!(concatenate_axis(Axis::axis(1), (
            tf32!([[1., 2.]]),
            tf32!([[10., 20.]])
        )), tf32!([
            [1., 2., 10., 20.],
        ]));

        assert_eq!(concatenate_axis(Axis::axis(1), (
            tf32!([[1., 2.], [3., 4.]]),
            tf32!([[10.], [20.]])
        )), tf32!([
            [1., 2., 10.], [3., 4., 20.],
        ]));
    }
}
