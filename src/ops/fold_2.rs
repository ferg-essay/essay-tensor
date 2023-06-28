use crate::{Tensor, tensor::{TensorUninit, Shape}, model::{IntoForward}};

use crate::tensor::{Dtype};

pub trait BiFold<D:Dtype=f32> : Clone {
    fn apply(&self, acc: D, a: D, b: D) -> D;
}

impl<D:Dtype + Copy> Tensor<D> {
    pub fn bi_fold<Op: BiFold<D> + IntoForward>(
        &self, 
        init: D, 
        op: Op,
        b: &Self
    ) -> Tensor<D> {
        let len = self.broadcast_min(1, b, 1);
        let inner_size = self.cols();
        let batch = len / inner_size;

        let shape = self.shape();
        let o_shape: Shape = if shape.size() > 0 {
            shape.slice(1..)
        } else {
            Shape::scalar()
        };
    
        unsafe {
            let mut o_data = TensorUninit::<D>::new(len);

            let o_ptr = o_data.as_mut_ptr();

            for i in 0..batch {
                let a_ptr = self.as_wrap_ptr(i * inner_size);
                let b_ptr = b.as_wrap_ptr(i * inner_size);

                let mut acc = init;

                for j in 0..inner_size {
                    acc = op.apply(
                        acc, 
                        *a_ptr.add(j),
                        *b_ptr.add(j)
                    );
                }

                *o_ptr.add(i) = acc;
            }

            Tensor::from_uninit(o_data, &o_shape)
        }
    }
}
