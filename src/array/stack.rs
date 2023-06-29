use crate::{model::{Operation}, Tensor, prelude::{AxisOpt, Shape}, tensor::{TensorId, TensorUninit, Dtype, IntoTensorList}};

use super::axis::axis_from_rank;

pub fn stack<D>(x: impl IntoTensorList<D>, axis: impl Into<AxisOpt>) -> Tensor<D>
where
    D: Dtype + Clone
{
    let mut vec = Vec::<Tensor<D>>::new();

    x.into_list(&mut vec);

    stack_vec(vec, axis)
}

pub fn stack_vec<D>(x: Vec<Tensor<D>>, axis: impl Into<AxisOpt>) -> Tensor<D>
where
    D: Dtype + Clone
{
    let axis: AxisOpt = axis.into();

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
    let axis = axis.get_axis();

    let op = StackOp(axis);

    let x_ptr : Vec<&Tensor<D>> = x.iter().collect();

    //let node = NodeOp::new(x_ptr.as_slice(), Box::new(op.clone()));
    let id = TensorId::unset();

    let tensor = op.f(x_ptr.as_slice(), id);

    // Tape::set_tensor(tensor)
    tensor
}

impl<D: Dtype + Clone> Tensor<D> {
    pub fn stack(&self, others: impl IntoTensorList<D>, axis: impl Into<AxisOpt>) -> Tensor<D> {
        let mut vec = Vec::<Tensor<D>>::new();
        vec.push(self.clone());

        others.into_list(&mut vec);

        stack_vec(vec, axis)
    }
}

#[derive(Clone)]
pub struct StackOp(Option<isize>);

impl StackOp {
    fn axis(&self) -> &Option<isize> {
        &self.0
    }
}

impl<D: Dtype + Clone> Operation<D> for StackOp {
    fn f(
        &self,
        args: &[&Tensor<D>],
        id: TensorId,
    ) -> Tensor<D> {
        let shape = args[0].shape();

        let axis = axis_from_rank(self.axis(), shape.rank() + 1);

        let n_args = args.len();
        let x_len = shape.size();
        let n_inner = shape.sublen(axis..);
        let n_outer = x_len / n_inner;

        let o_len = args.iter().map(|t| t.len()).sum();

        unsafe {
            let mut out = TensorUninit::<D>::new(o_len);

            let o = out.as_mut_slice();

            for (j, x) in args.iter().enumerate() {
                assert_eq!(x_len, x.len());

                let x = x.as_slice();

                for k in 0..n_outer {
                    for i in 0..n_inner {
                        o[(k * n_args + j) * n_inner + i] = x[k * n_inner + i].clone();
                    }
                }
            }

            let mut vec = Vec::from(args[0].shape().as_slice());
            vec.insert(axis, args.len());
    
            out.into_tensor_with_id(vec, id)
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{prelude::*, array::{stack, Axis}};
    
    #[test]
    fn test_stack() {
        assert_eq!(stack(vec![
            tf32!([1.]),
            tf32!([10.])
        ], ()), tf32!([
            [1.], 
            [10.]
        ]));

        assert_eq!(stack(vec![
            tf32!([1., 2., 3., 4.]),
            tf32!([10., 20., 30., 40.])
        ], ()), tf32!([
            [1., 2., 3., 4.],
            [10., 20., 30., 40.]
        ]));
    }
    
    #[test]
    fn test_stack_axis() {
        assert_eq!(stack(vec![
            tf32!([1.]),
            tf32!([10.])
        ], Axis::axis(-1)), tf32!([
            [1., 10.], 
        ]));

        assert_eq!(stack(vec![
            tf32!([1., 2., 3., 4.]),
            tf32!([10., 20., 30., 40.])
        ], Axis::axis(-1)), tf32!([
            [1., 10.],
            [2., 20.],
            [3., 30.],
            [4., 40.],
        ]));
    }
}
