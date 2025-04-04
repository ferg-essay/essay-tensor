use crate::{
    prelude::{Axis, Shape}, tensor::{Type, IntoTensorList, Tensor, TensorData},
};

// use super::axis::axis_from_rank;

pub fn stack<D>(x: impl IntoTensorList<D>) -> Tensor<D>
where
    D: Type + Clone
{
    let mut vec = Vec::<Tensor<D>>::new();

    x.into_list(&mut vec);

    stack_vec(vec, None)
}

pub fn stack_axis<D>(axis: impl Into<Axis>, x: impl IntoTensorList<D>) -> Tensor<D>
where
    D: Type + Clone
{
    let mut vec = Vec::<Tensor<D>>::new();

    x.into_list(&mut vec);

    stack_vec(vec, axis)
}

pub fn stack_vec<D>(x: Vec<Tensor<D>>, axis: impl Into<Axis>) -> Tensor<D>
where
    D: Type + Clone
{
    let axis: Axis = axis.into();

    let mut shape : Option<Shape> = None;
    for x in &x {
        shape = match shape {
            None => Some(x.shape().clone()),
            Some(shape) => { 
                assert_eq!(&shape, x.shape(), "stack() tensor shape must match");
                Some(shape)
            }
        }
    }
    // TODO: issues with batch and initial size not matching
    // let axis = axis.get_axis();

    let shape = x[0].shape();

    let axis = axis.axis_from_rank(shape.rank() + 1);

    let n_args = x.len();
    let x_len = shape.size();
    let n_inner = shape.sublen(axis, shape.rank());
    let n_outer = x_len / n_inner;

    let o_len = x.iter().map(|t| t.size()).sum();

    unsafe {
        TensorData::<D>::unsafe_init(o_len, |o| {
            for (j, x) in x.iter().enumerate() {
                assert_eq!(x_len, x.size());

                let x = x.as_slice();

                for k in 0..n_outer {
                    for i in 0..n_inner {
                        o.add((k * n_args + j) * n_inner + i)
                            .write(x[k * n_inner + i].clone());
                    }
                }
            }
        }).into_tensor(x[0].shape().clone().insert(axis, x.len()))
    }
}

impl<D: Type + Clone> Tensor<D> {
    pub fn stack(&self, others: impl IntoTensorList<D>, axis: impl Into<Axis>) -> Tensor<D> {
        let mut vec = Vec::<Tensor<D>>::new();
        vec.push(self.clone());

        others.into_list(&mut vec);

        stack_vec(vec, axis)
    }
}

#[cfg(test)]
mod test {
    use crate::{prelude::*, array::{stack, stack_axis}};
    
    #[test]
    fn test_stack() {
        assert_eq!(stack(vec![
            tf32!([1.]),
            tf32!([10.])
        ]), tf32!([
            [1.], 
            [10.]
        ]));

        assert_eq!(stack(vec![
            tf32!([1., 2., 3., 4.]),
            tf32!([10., 20., 30., 40.])
        ]), tf32!([
            [1., 2., 3., 4.],
            [10., 20., 30., 40.]
        ]));
    }
    
    #[test]
    fn test_stack_axis() {
        assert_eq!(stack_axis(Axis::axis(-1), vec![
            tf32!([1.]),
            tf32!([10.])
        ]), tf32!([
            [1., 10.], 
        ]));

        assert_eq!(stack_axis(Axis::axis(-1), vec![
            tf32!([1., 2., 3., 4.]),
            tf32!([10., 20., 30., 40.])
        ]), tf32!([
            [1., 10.],
            [2., 20.],
            [3., 30.],
            [4., 40.],
        ]));
    }
}
