use std::{cmp};

use crate::{
    Tensor, 
    tensor::{TensorUninit, TensorId}, 
    model::{NodeOp, Tape, Operation, Graph}, prelude::Shape
};

#[derive(Clone, Copy, PartialEq)]
pub struct LinspaceCpu(usize);

pub fn linspace(start: impl Into<Tensor>, end: impl Into<Tensor>, len: usize) -> Tensor {
    let start = start.into();
    let end = end.into();

    assert_eq!(start.shape(), end.shape(), 
        "linspace shapes must agree start={:?} end={:?}",
        start.shape(), end.shape());

    let linspace_op = LinspaceCpu(len);

    let id = NodeOp::new(&[&start, &end], Box::new(linspace_op));

    let tensor = linspace_op.f(&[&start, &end], id);

    Tape::set_tensor(tensor)
}

impl Tensor {
    pub fn linspace(&self, end: &Tensor, len: usize) -> Tensor {
        linspace(self, end, len)
    }
}

impl LinspaceCpu {
    #[inline]
    fn len(&self) -> usize {
        self.0
    }
}

impl Operation for LinspaceCpu {
    fn f(
        &self,
        args: &[&Tensor],
        id: TensorId,
    ) -> Tensor {
        assert!(args.len() == 2);

        let start = args[0];
        let end = args[1];

        assert_eq!(start.shape(), end.shape());

        let batch = cmp::max(1, start.len());
        let len = self.len();
        let size = batch * len;

        let mut o_shape_vec = Vec::from(start.shape().as_slice());
        o_shape_vec.insert(0, len);
        let o_shape = Shape::from(o_shape_vec);

        unsafe {
            let mut o_data = TensorUninit::<f32>::new(size);

            let o = o_data.as_mut_ptr();
            for n in 0..batch {
                let start = start[n];
                let end = end[n];

                assert!(start <= end);

                let step = if start < end && len > 0 {
                    (end - start) / (len - 1) as f32
                } else {
                    0.
                };

                for k in 0..len {
                    *o.add(k * batch + n) = start + step * k as f32;
                }
            }

            Tensor::from_uninit_with_id(o_data, o_shape, id)
        }
    }

    fn df(
        &self,
        _forward: &Graph,
        _graph: &mut Graph,
        _i: usize,
        _args: &[TensorId],
        _prev: TensorId,
    ) -> TensorId {
        todo!();
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;
    use crate::{init::linspace::linspace, tf32};

    #[test]
    fn linspace_0_4_5() {
        assert_eq!(linspace(0., 1., 2), tf32!([0., 1.]));
        assert_eq!(linspace(0., 4., 5), tf32!([0., 1., 2., 3., 4.]));

        assert_eq!(linspace(0., 4., 0), Tensor::<f32>::from(Vec::<f32>::new()));
    }

    #[test]
    fn linspace_vector_n() {
        assert_eq!(linspace([0.], [1.], 2), tf32!([[0.], [1.]]));
        assert_eq!(
            linspace([0., 0., 0.], [1., 2., 3.], 2), 
            tf32!([[0., 0., 0.], [1., 2., 3.]])
        );
    }

    #[test]
    fn linspace_tensor() {
        assert_eq!(linspace([[0.]], [[1.]], 2), tf32!([[[0.]], [[1.]]]));
        assert_eq!(
            linspace([[0., 0., 0.], [0., 0., 0.]], [[1., 2., 3.], [10., 20., 30.]], 3), 
            tf32!([
                [[0.0, 0.0, 0.0], [0., 0., 0.,]],
                [[0.5, 1.0, 1.5], [5., 10., 15.,]],
                [[1.0, 2.0, 3.0], [10., 20., 30.]]
            ])
        );
    }
}