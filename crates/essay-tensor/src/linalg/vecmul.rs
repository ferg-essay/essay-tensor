use crate::{Tensor, tensor::TensorUninit, model::{IntoForward, BoxForwardOp, Graph, TensorId, ForwardOp}};


#[derive(Debug, Clone)]
struct InnerProduct;

#[derive(Debug, Clone)]
struct OuterProduct;


impl Tensor<f32> {
    pub fn outer_product(&self, b: &Tensor<f32>) -> Tensor {
        outer_product(&self, b)
    }
}

pub fn outer_product(
    a: &Tensor<f32>,
    b: &Tensor<f32>,
) -> Tensor<f32> {
    assert!(a.rank() >= 1, "vector outer product dim[{}] should be >= 1", a.rank());
    assert_eq!(a.rank(), b.rank(), "vector outer product rank must match");
    assert_eq!(a.shape()[1..], b.shape()[1..], "outer product shape must match");

    let n : usize = a.shape()[1..].iter().product();

    let a_cols = a.shape()[0];
    let b_cols = b.shape()[0];

    let o_cols = a_cols;
    let o_rows = b_cols;
    let o_size = o_cols * o_rows;

    unsafe {
        let mut out = TensorUninit::<f32>::new(o_size * n);

        for block in 0..n {
            naive_outer_product_f32(
                &mut out, 
                block * o_size,
                o_cols,
                o_rows,
                &a,
                block * a_cols,
                &b,
                block * b_cols,
            );
        }

        let mut o_shape = vec![o_cols, o_rows];
        for size in &a.shape()[1..] {
            o_shape.push(*size);
        }

        // Tensor::new(Rc::new(out), o_shape)
        a.next_binop(&b, out.init(), o_shape, OuterProduct)
        //todo!()
    }
}

unsafe fn naive_outer_product_f32(
    out: &mut TensorUninit<f32>, 
    out_start: usize,
    o_cols: usize,
    o_rows: usize,
    a: &Tensor<f32>, 
    a_start: usize,
    b: &Tensor<f32>,
    b_start: usize,
) {
    let a_data = a.data();
    let b_data = b.data();

    // let mut a_row = a_start;

    for row in 0..o_rows {
        for col in 0..o_cols {
            let v = a_data.get_unchecked(a_start + col)
                    * b_data.get_unchecked(b_start + row);

            out.set_unchecked(out_start + row * o_cols + col, v);
        }
    }
}

impl ForwardOp for OuterProduct {
    fn eval(
        &self,
        tensors: &crate::model::TensorCache,
        args: &[&Tensor],
    ) -> Tensor {
        todo!()
    }

    fn backtrace(
        &self,
        forward: &Graph,
        graph: &mut Graph,
        i: usize,
        args: &[TensorId],
        out: TensorId,
        prev: TensorId,
    ) -> TensorId {
        todo!();
    }

    fn backtrace_top(
        &self,
        forward: &Graph,
        graph: &mut Graph,
        i: usize,
        args: &[TensorId],
        tensor: TensorId,
    ) -> TensorId {
        todo!();
    }

    fn box_clone(&self) -> BoxForwardOp {
        todo!()
    }
}

impl IntoForward for OuterProduct {
    fn to_op(&self) -> BoxForwardOp {
        Box::new(self.clone())
    }
}
