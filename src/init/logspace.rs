use std::{cmp};

use crate::{
    Tensor, 
    tensor::{TensorUninit, TensorId}, 
    model::{Operation, Expr, expr::GradOperation}, prelude::Shape
};

#[derive(Clone, Copy, PartialEq)]
pub struct LogspaceCpu {
    len: usize,
    base: f32,
}

pub fn logspace(
    start: impl Into<Tensor>, 
    end: impl Into<Tensor>, 
    len: usize,
) -> Tensor {
    logspace_opt(start, end, len, ())
}

pub fn logspace_opt(
    start: impl Into<Tensor>, 
    end: impl Into<Tensor>, 
    len: usize,
    opt: impl Into<Opt>
) -> Tensor {
    let start = start.into();
    let end = end.into();

    assert_eq!(start.shape(), end.shape(), 
        "logspace shapes must agree start={:?} end={:?}",
        start.shape(), end.shape());

    let opt : Opt = opt.into();
    let base = match opt.base {
        Some(base) => base,
        None => 10.
    };

    let logspace_op = LogspaceCpu { base, len };

    //let id = NodeOp::new(&[&start, &end], Box::new(linspace_op));
    let id = TensorId::unset();

    let tensor = logspace_op.f(&[&start, &end], id);

    //Tape::set_tensor(tensor)
    tensor
}

impl Tensor {
    pub fn logspace(&self, end: &Tensor, len: usize) -> Tensor {
        logspace(self, end, len)
    }
}

impl Operation<f32> for LogspaceCpu {
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
        let len = self.len;
        let base = self.base;
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

                let step = if len > 1 {
                    (end - start) / (len - 1) as f32
                } else {
                    0.
                };

                for k in 0..len {
                    let v = start + step * k as f32;
                    *o.add(k * batch + n) = base.powf(v);
                }
            }

            o_data.into().into_tensor(o_shape).with_id(id)
        }
    }
}

impl GradOperation<f32> for LogspaceCpu {
        fn df(
        &self,
        _forward: &Expr,
        _graph: &mut Expr,
        _i: usize,
        _args: &[TensorId],
        _prev: TensorId,
    ) -> TensorId {
        todo!();
    }
}

pub struct Opt {
    base: Option<f32>,
}

impl Default for Opt {
    fn default() -> Self {
        Self { base: Default::default() }
    }
}

impl From<()> for Opt {
    fn from(_value: ()) -> Self {
        Opt::default()
    }
}

impl From<f32> for Opt {
    fn from(base: f32) -> Self {
        Opt { base: Some(base) }
    }
}

#[cfg(test)]
mod test {
    use crate::init::logspace_opt;
    use crate::{init::logspace::logspace, tf32};

    #[test]
    fn logspace_0_2_3() {
        assert_eq!(logspace(0., 2., 3), tf32!([1., 10., 100.]));
    }

    #[test]
    fn logspace_opt_0_2_3() {
        assert_eq!(logspace_opt(0., 2., 3, 2.), tf32!([1., 2., 4.]));
    }
}