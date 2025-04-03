use std::cmp;

use crate::{
    Tensor, 
    tensor::TensorUninit, 
    prelude::Shape
};

#[derive(Clone, Copy, PartialEq)]
pub struct GeomspaceCpu {
    len: usize,
}

pub fn geomspace(
    start: impl Into<Tensor>, 
    end: impl Into<Tensor>, 
    len: usize,
) -> Tensor {
    let start = start.into();
    let end = end.into();

    assert_eq!(start.shape(), end.shape(), 
        "geomspace shapes must agree start={:?} end={:?}",
        start.shape(), end.shape());

    let geomspace_op = GeomspaceCpu { len };

    //let id = NodeOp::new(&[&start, &end], Box::new(linspace_op));
    //let id = TensorId::unset();

    let tensor = geomspace_op.f(&[&start, &end]);

    //Tape::set_tensor(tensor)
    tensor
}

impl Tensor {
    pub fn geomspace(&self, end: &Tensor, len: usize) -> Tensor {
        geomspace(self, end, len)
    }
}

//impl Operation<f32> for GeomspaceCpu {
impl GeomspaceCpu {
    fn f(
        &self,
        args: &[&Tensor],
    ) -> Tensor {
        assert!(args.len() == 2);

        let start = args[0];
        let end = args[1];

        assert_eq!(start.shape(), end.shape());

        let batch = cmp::max(1, start.len());
        let len = self.len;
        let size = batch * len;

        //let start_ln = start.ln();
        //let end_ln = end.ln();

        let mut o_shape_vec = Vec::from(start.shape().as_vec());
        o_shape_vec.insert(0, len);
        let o_shape = Shape::from(o_shape_vec);

        unsafe {
            let mut o_data = TensorUninit::<f32>::new(size);

            let o = o_data.as_mut_ptr();
            for n in 0..batch {
                let start = start[n].ln();
                let end = end[n].ln();

                assert!(start <= end);

                let step = if len > 1 {
                    (end - start) / (len - 1) as f32
                } else {
                    0.
                };

                for k in 0..len {
                    let v = start + step * k as f32;
                    *o.add(k * batch + n) = v.exp();
                }
            }

            let o_data = o_data.into();

            o_data.into_tensor(o_shape)
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{init::{geomspace}, tf32};

    #[test]
    fn geomspace_1_4_3() {
        assert_eq!(geomspace(1., 4., 3), tf32!([1., 2., 4.]));
        assert_eq!(geomspace(1., 8., 4), tf32!([1., 2., 4., 8.]));
        assert_eq!(
            geomspace(1., 8., 7),
            tf32!([1., 1.4142135, 2., 2.828427, 4., 5.656854, 8.]
        ));
    }
}