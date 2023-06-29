use crate::{model::{Operation}, Tensor, prelude::{AxisOpt, Shape}, tensor::{TensorId, TensorUninit, Dtype}};

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

    //pub fn append(&self, others: &[Tensor<D>], axis: impl Into<AxisOpt>) -> Tensor<D> {
    //    self.concatenate(others, axis)
    //}
}

pub trait IntoTensorList<D: Dtype> {
    fn into_list(self, vec: &mut Vec<Tensor<D>>);
}

impl<D: Dtype> IntoTensorList<D> for Vec<Tensor<D>> {
    fn into_list(self, vec: &mut Vec<Tensor<D>>) {
        let mut this = self;

        vec.append(&mut this)
    }
}

impl<D: Dtype> IntoTensorList<D> for &[Tensor<D>] {
    fn into_list(self, vec: &mut Vec<Tensor<D>>) {
        let mut vec2 = Vec::from(self);
        vec.append(&mut vec2);
    }
}

impl<D: Dtype, const N: usize> IntoTensorList<D> for [Tensor<D>; N] {
    fn into_list(self, vec: &mut Vec<Tensor<D>>) {
        let mut vec2 = Vec::from(self);
        vec.append(&mut vec2);
    }
}

macro_rules! tensor_list {
    ($($id:ident),*) => {
        #[allow(non_snake_case)]
        impl<D: Dtype, $($id),*> IntoTensorList<D> for ($($id,)*) 
        where $(
            $id: Into<Tensor<D>>
        ),*
        {
            fn into_list(self, vec: &mut Vec<Tensor<D>>) {
                let ($($id,)*) = self;

                $(
                    vec.push($id.into())
                );*
            }
        }
    }
}

tensor_list!(P0);
tensor_list!(P0, P1);
tensor_list!(P0, P1, P2);
tensor_list!(P0, P1, P2, P3);
tensor_list!(P0, P1, P2, P3, P4);
tensor_list!(P0, P1, P2, P3, P4, P5);
tensor_list!(P0, P1, P2, P3, P4, P5, P6);
tensor_list!(P0, P1, P2, P3, P4, P5, P6, P7);
tensor_list!(P0, P1, P2, P3, P4, P5, P6, P7, P8);
tensor_list!(P0, P1, P2, P3, P4, P5, P6, P7, P8, P9);

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

        let shape_inner = shape.remove(axis);

        //let o_len = shape_inner.size() * axis_len;

        let n_args = args.len();
        let x_len = shape.size();
        let n_inner = shape.sublen(axis + 1..);
        let n_outer = shape.sublen(0..axis);
        let n_common = n_inner * n_outer;

        let o_len = args.iter().map(|t| t.len()).sum();

        unsafe {
            let mut out = TensorUninit::<D>::new(o_len);

            let mut o = out.as_mut_ptr();

            for (n, x) in args.iter().enumerate() {
                let x = x.as_slice();
                //let n_axis = x.len() / n_common;

                // TODO: axis
                /*
                for k in 0..n_outer {
                    for j in 0..n_axis {
                        for i in 0..n_inner {
                            o[(k * n_args + j) * n_inner + i] = x[k * n_inner + i].clone();
                        }
                    }
                }
                */
                for i in 0..x.len() {
                    *o.add(i) = x[i].clone();
                }

                o = o.add(x.len());
            }

            let shape = shape_inner.insert(axis, axis_len);
    
            out.into_tensor_with_id(shape, id)
        }
    }
}

fn axis_from_rank(axis: &Option<isize>, rank: usize) -> usize {
    match axis {
        Some(axis) => (axis + rank as isize) as usize % rank,
        None => 0,
    }
}

#[cfg(test)]
mod test {
    use crate::{prelude::*, array::{stack, Axis, concatenate}};
    
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
    }
}
