use crate::{model::{Operation}, Tensor, prelude::{AxisOpt}, tensor::{TensorId, TensorUninit, Dtype, IntoTensorList}};

pub fn concatenate<D>(x: impl IntoTensorList<D>, axis: impl Into<AxisOpt>) -> Tensor<D>
where
    D: Dtype + Clone
{
    let mut vec = Vec::<Tensor<D>>::new();

    x.into_list(&mut vec);

    concat_vec(vec, axis)
}

pub fn concat_vec<D>(x: Vec<Tensor<D>>, axis: impl Into<AxisOpt>) -> Tensor<D>
where
    D: Dtype + Clone
{
    let axis: AxisOpt = axis.into();

    let op = ConcatOp(axis.get_axis());

    let x_ptr : Vec<&Tensor<D>> = x.iter().collect();

    //let node = NodeOp::new(x_ptr.as_slice(), Box::new(op.clone()));
    let id = TensorId::unset();

    let tensor = op.f(x_ptr.as_slice(), id);

    D::set_tape(tensor)
}

impl<D: Dtype + Clone> Tensor<D> {
    pub fn concat(
        &self, others: impl IntoTensorList<D>, 
        axis: impl Into<AxisOpt>
    ) -> Tensor<D> {
        let mut vec = Vec::<Tensor<D>>::new();
        vec.push(self.clone());

        others.into_list(&mut vec);

        concat_vec(vec, axis)
    }
}

#[derive(Clone)]
pub struct ConcatOp(Option<isize>);

impl ConcatOp {
    fn axis(&self) -> &Option<isize> {
        &self.0
    }
}

impl<D: Dtype + Clone> Operation<D> for ConcatOp {
    fn f(
        &self,
        args: &[&Tensor<D>],
        id: TensorId,
    ) -> Tensor<D> {

        let shape = args[0].shape().clone();

        let axis = axis_from_rank(self.axis(), shape.rank());

        concat_impl(args, axis, id)
    }
}

pub(crate) fn concat_impl<D: Dtype + Clone>(
    args: &[&Tensor<D>], 
    axis: usize, 
    id: TensorId,
) -> Tensor<D> {
    let axis_len = validate_concat(args, axis);

    let shape = args[0].shape();

    let shape_inner = shape.remove(axis);

    let o_len = args.iter().map(|t| t.len()).sum();

    unsafe {
        let mut out = TensorUninit::<D>::new(o_len);

        let o = out.as_mut_ptr();

        let n_outer = shape_inner.sublen(0..axis);
        let n_inner = if axis < shape_inner.rank() {
            shape_inner.sublen(axis..)
        } else {
            1
        };

        let j_stride = n_inner * axis_len;
        let mut t = 0;

        for tensor in args.iter() {
            let n_axis = tensor.shape().dim(axis);

            let x = tensor.as_slice();

            for j in 0..n_outer {
                for k in 0..n_axis {
                    for i in 0..n_inner {
                        *o.add(j * j_stride + t + k * n_inner + i)
                            = x[j * n_axis * n_inner + k * n_inner + i].clone();
                    }
                }
            }

            t += n_axis * n_inner;
        }

        let shape = shape_inner.insert(axis, axis_len);

        out.into().into_tensor(shape).with_id(id)
    }
}

fn validate_concat<D>(args: &[&Tensor<D>], axis: usize) -> usize {
    let shape = args[0].shape();

    let mut axis_len = 0;

    // TODO: split out validation
    for x in args {
        let x_shape = x.shape();
    
        assert_eq!(shape.rank(), x_shape.rank(),
            "concat() tensor shape must match {:?} {:?}", 
            shape.as_slice(), x_shape.as_slice()
        );
    
        for i in 0..shape.rank() {
            if i == axis {
                axis_len += x_shape.dim(axis);
            } else {
                assert_eq!(x_shape.dim(i), shape.dim(i),
                    "concat() tensor shape must match {:?} {:?}", 
                    shape.as_slice(), x_shape.as_slice()
                );
            }
        }
    }

    axis_len
}

fn axis_from_rank(axis: &Option<isize>, rank: usize) -> usize {
    match axis {
        Some(axis) => (axis + rank as isize) as usize % rank,
        None => 0,
    }
}

#[cfg(test)]
mod test {
    use crate::{prelude::*, array::{Axis, concatenate}};
    
    #[test]
    fn test_concat() {
        assert_eq!(concatenate((
            tf32!([1.]),
            tf32!([10.])
        ), ()), tf32!([1., 10.]));

        assert_eq!(concatenate((
            tf32!([1., 2.]),
            tf32!([10., 20., 30.])
        ), ()), tf32!([
            1., 2., 10., 20., 30.,
        ]));

        assert_eq!(concatenate((
            tf32!([[1.], [2.]]),
            tf32!([[10.], [20.], [30.]])
        ), ()), tf32!([
            [1.], [2.], [10.], [20.], [30.],
        ]));

        assert_eq!(concatenate((
            tf32!([[1., 2.]]),
            tf32!([[10., 20.], [30., 40.]])
        ), ()), tf32!([
            [1., 2.], [10., 20.], [30., 40.],
        ]));

        assert_eq!(concatenate((
            tf32!([[[1., 2.]]]),
            tf32!([[[10., 20.]], [[30., 40.]]])
        ), ()), tf32!([
            [[1., 2.]], [[10., 20.]], [[30., 40.]],
        ]));
    }
    
    #[test]
    fn test_concat_axis() {
        assert_eq!(concatenate((
            tf32!([1.]),
            tf32!([10.])
        ), Axis::axis(0)), tf32!([
            1., 10., 
        ]));

        assert_eq!(concatenate((
            tf32!([1., 2.]),
            tf32!([10., 20., 30.])
        ), Axis::axis(-1)), tf32!([
            1., 2., 10., 20., 30.,
        ]));

        assert_eq!(concatenate((
            tf32!([[1., 2.]]),
            tf32!([[10., 20.]])
        ), Axis::axis(0)), tf32!([
            [1., 2.], [10., 20.],
        ]));

        assert_eq!(concatenate((
            tf32!([[1., 2.]]),
            tf32!([[10., 20.]])
        ), Axis::axis(1)), tf32!([
            [1., 2., 10., 20.],
        ]));

        assert_eq!(concatenate((
            tf32!([[1., 2.], [3., 4.]]),
            tf32!([[10.], [20.]])
        ), Axis::axis(1)), tf32!([
            [1., 2., 10.], [3., 4., 20.],
        ]));
    }
}
