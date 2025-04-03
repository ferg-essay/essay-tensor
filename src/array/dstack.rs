use crate::{
    Tensor, 
    prelude::Shape, 
    tensor::{TensorId, Dtype, IntoTensorList}
};

pub fn dstack<D>(x: impl IntoTensorList<D>) -> Tensor<D>
where
    D: Dtype + Clone
{
    let mut vec = Vec::<Tensor<D>>::new();

    x.into_list(&mut vec);

    dstack_vec(vec)
}

pub fn dstack_vec<D>(x: Vec<Tensor<D>>) -> Tensor<D>
where
    D: Dtype + Clone
{
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

    let op = DstackOp;

    let x_ptr : Vec<&Tensor<D>> = x.iter().collect();

    //let node = NodeOp::new(x_ptr.as_slice(), Box::new(op.clone()));
    let id = TensorId::unset();

    //let tensor = op.f(x_ptr.as_slice(), id);

    // Tape::set_tensor(tensor)
    //tensor
    todo!();
}

impl<D: Dtype + Clone> Tensor<D> {
    pub fn dstack(&self, others: impl IntoTensorList<D>) -> Tensor<D> {
        let mut vec = Vec::<Tensor<D>>::new();
        vec.push(self.clone());

        others.into_list(&mut vec);

        dstack_vec(vec)
    }
}

#[derive(Clone)]
pub struct DstackOp;

/*
impl<D: Dtype + Clone> Operation<D> for DstackOp {
    fn f(
        &self,
        args: &[&Tensor<D>],
        id: TensorId,
    ) -> Tensor<D> {
        let tensors : Vec<Tensor<D>> = args.iter().map(|x| {
            match x.shape().rank() {
                0 => todo!(),
                1 => x.reshape([1, x.len(), 1]),
                2 => x.reshape([x.rows(), x.cols(), 1]),
                _ => (*x).clone(),
            }
        }).collect();

        let vec : Vec<&Tensor<D>> = tensors.iter().collect();

        concat_impl(vec.as_slice(), 2, id)
    }
}
    */

#[cfg(test)]
mod test {
    use crate::{prelude::*, array::{dstack}};
    
    #[test]
    fn test_dstack() {
        assert_eq!(dstack(vec![
            tf32!([1.]),
            tf32!([10.])
        ]), tf32!([
            [[1., 10.]], 
        ]));

        assert_eq!(dstack(vec![
            tf32!([1., 2.]),
            tf32!([10., 20.])
        ]), tf32!([
            [[1., 10.],
            [2., 20.]]
        ]));

        assert_eq!(dstack(vec![
            tf32!([[1.], [2.], [3.]]),
            tf32!([[10.], [20.], [30.]])
        ]), tf32!([
            [[1., 10.]],
            [[2., 20.]],
            [[3., 30.]],
        ]));
    }
}
