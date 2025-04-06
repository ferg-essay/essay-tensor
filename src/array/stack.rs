use crate::tensor::{Axis, Shape, Type, IntoTensorList, Tensor, unsafe_init};

pub fn concatenate<T>(x: impl IntoTensorList<T>) -> Tensor<T>
where
    T: Type + Clone
{
    let mut vec = Vec::<Tensor<T>>::new();

    x.into_list(&mut vec);

    concat_axis(0, vec)
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

    concat_axis(axis, vec)
}

// use super::axis::axis_from_rank;

pub fn stack<T>(x: impl IntoTensorList<T>) -> Tensor<T>
where
    T: Type + Clone
{
    let mut vec = Vec::<Tensor<T>>::new();

    x.into_list(&mut vec);

    stack_vec(None, vec)
}

pub fn stack_axis<T>(axis: impl Into<Axis>, x: impl IntoTensorList<T>) -> Tensor<T>
where
    T: Type + Clone
{
    let mut vec = Vec::<Tensor<T>>::new();

    x.into_list(&mut vec);

    stack_vec(axis, vec)
}

pub fn dstack<T>(x: impl IntoTensorList<T>) -> Tensor<T>
where
    T: Type + Clone
{
    let mut vec = Vec::<Tensor<T>>::new();

    x.into_list(&mut vec);

    dstack_vec(vec)
}

pub fn hstack<D>(x: impl IntoTensorList<D>) -> Tensor<D>
where
    D: Type + Clone
{
    let mut vec = Vec::<Tensor<D>>::new();

    x.into_list(&mut vec);

    let shape = vec[0].shape();

    if shape.rank() == 1 {
        concatenate_axis(0, vec)
    } else {
        concatenate_axis(1, vec)
    }
}

pub fn vstack<T>(x: impl IntoTensorList<T>) -> Tensor<T>
where
    T: Type + Clone
{
    let mut vec = Vec::<Tensor<T>>::new();

    x.into_list(&mut vec);

    vstack_vec(vec)
}

impl<D: Type + Clone> Tensor<D> {
}

impl<T: Type + Clone> Tensor<T> {
    pub fn concatenate(&self, others: impl IntoTensorList<T>) -> Tensor<T> {
        let mut vec = Vec::<Tensor<T>>::new();
        vec.push(self.clone());

        others.into_list(&mut vec);

        concatenate(vec)
    }

    pub fn concatenate_axis(
        &self, axis: impl Into<Axis>, 
        others: impl IntoTensorList<T>
    ) -> Tensor<T> {
        let mut vec = Vec::<Tensor<T>>::new();
        vec.push(self.clone());

        others.into_list(&mut vec);

        concatenate_axis(axis, vec)
    }

    pub fn stack(&self, others: impl IntoTensorList<T>, axis: impl Into<Axis>) -> Tensor<T> {
        let mut vec = Vec::<Tensor<T>>::new();
        vec.push(self.clone());

        others.into_list(&mut vec);

        stack_vec(axis, vec)
    }

    pub fn dstack(&self, others: impl IntoTensorList<T>) -> Tensor<T> {
        let mut vec = Vec::<Tensor<T>>::new();
        vec.push(self.clone());

        others.into_list(&mut vec);

        dstack_vec(vec)
    }

    pub fn vstack(
        &self, others: impl IntoTensorList<T>, 
    ) -> Tensor<T> {
        let mut vec = Vec::<Tensor<T>>::new();
        vec.push(self.clone());

        others.into_list(&mut vec);

        vstack_vec(vec)
    }

    #[inline]
    pub fn unstack(&self, axis: impl Into<Axis>) -> Vec<Tensor<T>> {
        let axis : Axis = axis.into();
    
        let axis = axis.axis_from_rank(self.rank());
    
        let cuts = Vec::<usize>::new();
    
        if axis == 0 {
            let mut slices = Vec::<Tensor<T>>::new();
    
            let mut prev = 0;
            for i in cuts {
                if i != prev {
                    slices.push(self.subslice(prev, i - prev));
                }
                prev = i;
            }
        
    
            slices
        } else {
            self.split_axis(Axis::axis(axis as isize), cuts)
        }
    }
}

pub(crate) fn concat_axis<T: Type + Clone>(
    axis: usize,
    args: Vec<Tensor<T>>, 
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
        unsafe_init::<T>(o_len, shape, |o| {
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

        })
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

fn stack_vec<T>(axis: impl Into<Axis>, x: Vec<Tensor<T>>) -> Tensor<T>
where
    T: Type + Clone
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

    let shape = x[0].shape().clone().insert(axis, x.len());

    unsafe {
        unsafe_init::<T>(o_len, shape, |o| {
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
        })
    }
}

fn dstack_vec<T>(x: Vec<Tensor<T>>) -> Tensor<T>
where
    T: Type + Clone
{
    let shape = x[0].shape().clone();

    let tensors : Vec<Tensor<T>> = x.iter().map(|x| {
        assert_eq!(&shape, x.shape(), "stack() tensor shape must match");

        match x.shape().rank() {
            0 => todo!(),
            1 => x.clone().reshape([1, x.size(), 1]),
            2 => x.clone().reshape([x.rows(), x.cols(), 1]),
            _ => (*x).clone(),
        }
    }).collect();

    concatenate_axis(Axis::axis(2), tensors.as_slice())
}

pub fn vstack_vec<D>(x: Vec<Tensor<D>>) -> Tensor<D>
where
    D: Type + Clone
{
    let x_ptr : Vec<&Tensor<D>> = x.iter().collect();

    let expand_args : Vec<Tensor<D>> = x_ptr.iter().map(|t| {
        let shape = t.shape().clone().insert(0, 1);
        (*t).clone().reshape(shape)
    }).collect();

    //let vec : Vec<&Tensor<D>> = expand_args.iter().collect();

    concatenate_axis(Axis::axis(0), expand_args.as_slice()) // vec.as_slice())
}

#[cfg(test)]
mod test {
    use crate::{array::{concatenate, concatenate_axis, dstack, hstack, stack, stack_axis, vstack}, tensor::Axis, tf32};
    
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
    
    #[test]
    fn test_dstack() {
        assert_eq!(
            dstack(vec![
                tf32![1.],
                tf32![10.]
            ]), 
            tf32![
                [[1., 10.]] 
            ]
        );

        assert_eq!(dstack(vec![
            tf32!([1., 2.]),
            tf32!([10., 20.])
        ]), tf32!([
            [[1., 10.],
            [2., 20.]]
        ]));

        assert_eq!(dstack(vec![
            tf32!([[1.], [2.], [3.]]),
            tf32!([[10.], [20.], [30.]])
        ]), tf32!([
            [[1., 10.]],
            [[2., 20.]],
            [[3., 30.]],
        ]));
    }
    
    #[test]
    fn test_hstack() {
        assert_eq!(hstack((
            tf32!([1., 2.]),
            tf32!([10., 20., 30.])
        )), tf32!([1., 2., 10., 20., 30.]));

        assert_eq!(hstack((
            tf32!([[1., 2.]]),
            tf32!([[10., 20.]])
        )), tf32!([
            [1., 2., 10., 20.]
        ]));

        assert_eq!(hstack((
            tf32!([[1.], [2.]]),
            tf32!([[10.], [20.]])
        )), tf32!([
            [[1., 10.], [2., 20.]]
        ]));
    }
    #[test]
    fn test_vstack() {
        assert_eq!(vstack((
            tf32!([1.]),
            tf32!([10.])
        )), tf32!([[1.], [10.]]));

        assert_eq!(vstack((
            tf32!([[1., 2.]]),
            tf32!([[10., 20.]])
        )), tf32!([
            [[1., 2.]], [[10., 20.]]
        ]));
    }
}
