use crate::{
    tensor::{TensorId, Dtype, IntoTensorList},
    Tensor, 
};

pub fn hstack<D>(x: impl IntoTensorList<D>) -> Tensor<D>
where
    D: Dtype + Clone
{
    let mut vec = Vec::<Tensor<D>>::new();

    x.into_list(&mut vec);

    hstack_vec(vec)
}

pub fn hstack_vec<D>(x: Vec<Tensor<D>>) -> Tensor<D>
where
    D: Dtype + Clone
{
    let op = HstackOp;

    let x_ptr : Vec<&Tensor<D>> = x.iter().collect();

    //let node = NodeOp::new(x_ptr.as_slice(), Box::new(op.clone()));
    let id = TensorId::unset();

    // let tensor = op.f(x_ptr.as_slice(), id);

    // D::set_tape(tensor)

    todo!();
}

impl<D: Dtype + Clone> Tensor<D> {
    pub fn hstack(
        &self, others: impl IntoTensorList<D>, 
    ) -> Tensor<D> {
        let mut vec = Vec::<Tensor<D>>::new();
        vec.push(self.clone());

        others.into_list(&mut vec);

        hstack_vec(vec)
    }
}

#[derive(Clone)]
pub struct HstackOp;

/*
impl<D: Dtype + Clone> Operation<D> for HstackOp {
    fn f(
        &self,
        args: &[&Tensor<D>],
        id: TensorId,
    ) -> Tensor<D> {
        let shape = args[0].shape();

        if shape.rank() == 1 {
            concat_impl(args, 0, id)
        } else {
            concat_impl(args, 1, id)
        }
    }
}
    */

#[cfg(test)]
mod test {
    use crate::{prelude::*, array::{hstack}};
    
    #[test]
    fn test_hstack() {
        assert_eq!(hstack((
            tf32!([1., 2.]),
            tf32!([10., 20., 30.])
        )), tf32!([1., 2., 10., 20., 30.]));

        assert_eq!(hstack((
            tf32!([[1., 2.]]),
            tf32!([[10., 20.]])
        )), tf32!([
            [1., 2., 10., 20.]
        ]));

        assert_eq!(hstack((
            tf32!([[1.], [2.]]),
            tf32!([[10.], [20.]])
        )), tf32!([
            [[1., 10.], [2., 20.]]
        ]));
    }
}
