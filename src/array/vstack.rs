use crate::{Tensor, tensor::{Dtype, IntoTensorList}};

use super::concat::concat_axis;

pub fn vstack<D>(x: impl IntoTensorList<D>) -> Tensor<D>
where
    D: Dtype + Clone
{
    let mut vec = Vec::<Tensor<D>>::new();

    x.into_list(&mut vec);

    vstack_vec(vec)
}

pub fn vstack_vec<D>(x: Vec<Tensor<D>>) -> Tensor<D>
where
    D: Dtype + Clone
{
    let op = VstackOp;

    let x_ptr : Vec<&Tensor<D>> = x.iter().collect();

    //let node = NodeOp::new(x_ptr.as_slice(), Box::new(op.clone()));
    //let id = TensorId::unset();

    let tensor = op.f(x_ptr.as_slice());

    // D::set_tape(tensor)
    todo!();
}

impl<D: Dtype + Clone> Tensor<D> {
    pub fn vstack(
        &self, others: impl IntoTensorList<D>, 
    ) -> Tensor<D> {
        let mut vec = Vec::<Tensor<D>>::new();
        vec.push(self.clone());

        others.into_list(&mut vec);

        vstack_vec(vec)
    }
}

#[derive(Clone)]
pub struct VstackOp;

//impl<D: Dtype + Clone> Operation<D> for VstackOp {
impl VstackOp {
    fn f<D: Dtype + Clone>(
        &self,
        args: &[&Tensor<D>],
    ) -> Tensor<D> {
        let expand_args : Vec<Tensor<D>> = args.iter().map(|t| {
            let shape = t.shape().insert(0, 1);
            (*t).clone().with_shape(shape)
        }).collect();

        let vec : Vec<&Tensor<D>> = expand_args.iter().collect();

        //concat_axis(vec.as_slice(), 0)
        todo!()
    }
}

#[cfg(test)]
mod test {
    use crate::{prelude::*, array::{vstack}};
    
    #[test]
    fn test_vstack() {
        assert_eq!(vstack((
            tf32!([1.]),
            tf32!([10.])
        )), tf32!([[1.], [10.]]));

        assert_eq!(vstack((
            tf32!([[1., 2.]]),
            tf32!([[10., 20.]])
        )), tf32!([
            [[1., 2.]], [[10., 20.]]
        ]));
    }
}
