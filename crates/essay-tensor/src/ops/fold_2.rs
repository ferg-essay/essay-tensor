use crate::{Tensor, tensor::TensorUninit, graph::{IntoForward}};

use crate::tensor::{Dtype};

pub trait BiFold<D:Dtype=f32> : Clone {
    fn apply(&self, acc: D, a: D, b: D) -> D;
}

impl<D:Dtype> Tensor<D> {
    pub fn bi_fold<Op:BiFold<D> + IntoForward>(
        &self, 
        init: D, 
        op: Op,
        b: &Self
    ) -> Tensor<D> {
        let a_data = self.data();
        let b_data = b.data();

        let len = self.broadcast_min(1, b, 1);
        let inner_size = self.dim_zero();
        let batch = len / inner_size;
    
        let o_data = unsafe {
            let mut o_data = TensorUninit::<D>::new(len);

            let o_ptr = o_data.as_mut_ptr();

            for i in 0..batch {
                let a_ptr = a_data.as_wrap_ptr(i * inner_size);
                let b_ptr = b_data.as_wrap_ptr(i * inner_size);

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

            o_data.init()
        };
    
        let shape = self.shape();
        let o_shape: Vec<usize> = if shape.len() > 0 {
            shape[1..].iter().map(|d| *d).collect()
        } else {
            Vec::new()
        };

        Tensor::new(o_data, &o_shape)
    }
}
