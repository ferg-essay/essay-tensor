use crate::{Tensor, tensor::TensorUninit, model::{IntoForward}};

use crate::tensor::{Dtype};

pub trait BiFold<D:Dtype=f32> : Clone {
    fn apply(&self, state: D, a: D, b: D) -> D;
}

impl Tensor {
    pub fn bi_fold<Op:BiFold + IntoForward>(
        &self, 
        init: f32, 
        op: Op,
        b: &Self
    ) -> Tensor<f32> {
        assert_eq!(self.shape(), b.shape());
    
        let a_data = self.data();
        let b_data = b.data();

        let len = a_data.len();
        let stride = if self.rank() > 0 { self.dim(0) } else { 1 };
        let batch = len / stride;
    
        unsafe {
            let mut o_data = TensorUninit::<f32>::new(len);
    
            for i in 0..batch {
                let offset = i * stride;

                let mut state = init;

                for j in 0..stride {
                    state = op.apply(
                        state, 
                        a_data.get_unchecked(offset + j), 
                        b_data.get_unchecked(offset + j)
                    );
                }

                o_data.set_unchecked(i, state);
            }
    
            //Self::new(Rc::new(data), b.shape().clone())
    
            let shape = self.shape();
            let o_shape: Vec<usize> = if shape.len() > 0 {
                shape[1..].iter().map(|d| *d).collect()
            } else {
                Vec::new()
            };

            self.next_binop(&b, o_data.init(), o_shape, op)
        }
    }
}
