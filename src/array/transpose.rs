use crate::{
    Tensor, tensor::{TensorUninit, Dtype}
};

pub fn transpose<D: Dtype + Clone>(x: impl Into<Tensor<D>>) -> Tensor<D> {
    let op = Transpose;

    //let id = D::node_op(x, Box::new(op));
    // let id = TensorId::unset(); // D::node_op(x, Box::new(op));

    let x = x.into();

    let tensor = op.f(&[&x]);

    // D::set_tape(tensor)
    todo!();
}

impl<D: Dtype + Clone> Tensor<D> {
    #[inline]
    pub fn transpose(&self) -> Tensor<D> {
        transpose(self)
    }

    #[inline]
    pub fn t(&self) -> Tensor<D> {
        transpose(self)
    }
}

#[derive(Clone)]
pub struct Transpose;

impl Transpose {
    fn f<D: Dtype + Clone>(
        &self,
        args: &[&Tensor<D>],
    ) -> Tensor<D> {
        let tensor = args[0];

        let cols = tensor.cols().max(1);
        let rows = tensor.rows().max(1);
        let size = tensor.shape().size();
        let n_inner = cols * rows;
        let batch = size / n_inner;

        unsafe {
            let mut out = TensorUninit::<D>::new(size);

            let o = out.as_mut_slice();
            let x = tensor.as_slice();

            for n in 0..batch {
                for j in 0..rows {
                    for i in 0..cols {
                        o[n * n_inner + i * rows + j] = x[n * n_inner + j * cols + i].clone();
                    }
                }
            }

            let mut slice = Vec::from(tensor.shape().as_slice());
            slice.pop();
            slice.pop();
            slice.push(cols);
            slice.push(rows);
    
            Tensor::from_uninit(out, slice)
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{prelude::*, array::{transpose}};
    
    #[test]
    fn test_transpose() {
        assert_eq!(
            transpose(&tf32!([1., 2.])),
            tf32!([[1.], [2.]]),
        );

        assert_eq!(
            transpose(&tf32!([[1., 2.], [3., 4.], [5., 6.]])),
            tf32!([[1., 3., 5.], [2., 4., 6.]]),
        );
    }
}
