use std::{slice::SliceIndex, cmp, ops::Index};

use crate::{model::{Operation, Expr, NodeOp, Tape}, Tensor};

use super::{TensorId, TensorUninit, AxisOpt, Shape};


//
// stack operation
//

pub fn stack(x: &[Tensor], axis: impl Into<AxisOpt>) -> Tensor {
    let axis: AxisOpt = axis.into();

    let mut shape : Option<Shape> = None;
    for x in x {
        shape = match shape {
            None => Some(x.shape().clone()),
            Some(shape) => { 
                assert_eq!(&shape, x.shape(), "stack() tensor shape must match");
                Some(shape)
            }
        }
    }
    let shape = shape.unwrap();

    // TODO: issues with batch and initial size not matching
    let axis = axis.get_axis();

    let op = StackOp(axis);

    let x_ptr : Vec<&Tensor> = x.iter().collect();

    //let node = NodeOp::new(x_ptr.as_slice(), Box::new(op.clone()));
    let id = TensorId::unset();

    let tensor = op.f(x_ptr.as_slice(), id);

    // Tape::set_tensor(tensor)
    tensor
}

impl Tensor {
    pub fn stack(&self, others: &[Tensor], axis: impl Into<AxisOpt>) -> Tensor {
        let mut vec = Vec::<Tensor>::new();
        vec.push(self.clone());
        for x in others {
            vec.push(x.clone());
        }

        stack(&vec, axis)
    }
}

#[derive(Clone)]
pub struct StackOp(Option<isize>);

impl StackOp {
    fn axis(&self) -> &Option<isize> {
        &self.0
    }
}

impl Operation<f32> for StackOp {
    fn f(
        &self,
        args: &[&Tensor],
        id: TensorId,
    ) -> Tensor {
        let shape = args[0].shape();

        let axis = axis_from_rank(self.0, shape.rank() + 1);

        let n_args = args.len();
        let x_len = shape.size();
        let n_inner = shape.sublen(axis..);
        let n_outer = x_len / n_inner;

        let o_len = args.iter().map(|t| t.len()).sum();

        unsafe {
            let mut out = TensorUninit::<f32>::new(o_len);

            let o = out.as_mut_slice();

            for (j, x) in args.iter().enumerate() {
                assert_eq!(x_len, x.len());

                let x = x.as_slice();

                for k in 0..n_outer {
                    for i in 0..n_inner {
                        o[(k * n_args + j) * n_inner + i] = x[k * n_inner + i];
                    }
                }
            }

            let mut vec = Vec::from(args[0].shape().as_slice());
            vec.insert(axis, args.len());
    
            Tensor::from_uninit_with_id(out, vec, id)
        }
    }
}

fn axis_from_rank(axis: Option<isize>, rank: usize) -> usize {
    match axis {
        Some(axis) => (axis + rank as isize) as usize % rank,
        None => 0,
    }
}

#[cfg(test)]
mod test {
    use crate::{prelude::*, tensor::{stack, shape::Axis}};
    
    #[test]
    fn test_stack() {
        assert_eq!(stack(&vec![
            tf32!([1.]),
            tf32!([10.])
        ], ()), tf32!([
            [1.], 
            [10.]
        ]));

        assert_eq!(stack(&vec![
            tf32!([1., 2., 3., 4.]),
            tf32!([10., 20., 30., 40.])
        ], ()), tf32!([
            [1., 2., 3., 4.],
            [10., 20., 30., 40.]
        ]));
    }
    
    #[test]
    fn test_stack_axis() {
        assert_eq!(stack(&vec![
            tf32!([1.]),
            tf32!([10.])
        ], Axis::axis(-1)), tf32!([
            [1., 10.], 
        ]));

        assert_eq!(stack(&vec![
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
