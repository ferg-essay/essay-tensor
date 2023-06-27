use crate::{Tensor, tensor::{TensorId}, model::TensorCache};

use super::{Expr};

pub trait Tensors : Clone {
    fn fun_in(expr: &mut Expr, tensors: &mut TensorCache, input: &Self) -> Self;
    fn set_input(tensors: &mut TensorCache, index: usize, item: &Self) -> usize;

    fn out_ids(out: &mut Vec<TensorId>, item: &Self);
    fn make_out(cache: &TensorCache, out: &Vec<TensorId>, index: &mut usize) -> Self;
}

impl Tensors for Tensor<f32> {
    fn fun_in(expr: &mut Expr, tensors: &mut TensorCache, input: &Self) -> Self {
        let tensor = expr.arg(input.clone());

        let index = tensors.push(Some(tensor.clone()));
        assert_eq!(tensor.id().index(), index);

        tensor
    }

    fn set_input(out: &mut TensorCache, index: usize, item: &Self) -> usize {
        let id = out.new_id(index);
        
        out.set(id, item.clone().with_id(id));

        index + 1
    }


    fn out_ids(out: &mut Vec<TensorId>, item: &Self) {
        if item.id().is_some() {
            out.push(item.id())
        }
    }

    fn make_out(cache: &TensorCache, ids: &Vec<TensorId>, index: &mut usize) -> Self {
        let value = cache.get(ids[*index]).unwrap();
        *index += 1;
        value.clone()
    }
}

impl Tensors for () {
    fn fun_in(_expr: &mut Expr, _tensors: &mut TensorCache, _input: &Self) -> Self {
        ()
    }

    fn set_input(_out: &mut TensorCache, index: usize, _item: &Self) -> usize {
        index
    }

    fn out_ids(_out: &mut Vec<TensorId>, _item: &Self) {
    }

    fn make_out(_cache: &TensorCache, _ids: &Vec<TensorId>, _index: &mut usize) -> Self {
        ()
    }
}

impl<T: Tensors> Tensors for Vec<T> {
    fn fun_in(_expr: &mut Expr, _tensors: &mut TensorCache, _input: &Self) -> Self {
        todo!()
    }

    fn set_input(_out: &mut TensorCache, _index: usize, _item: &Self) -> usize {
        todo!()
    }

    fn out_ids(_out: &mut Vec<TensorId>, _item: &Self) {
        todo!()
    }

    fn make_out(_cache: &TensorCache, _ids: &Vec<TensorId>, _index: &mut usize) -> Self {
        todo!()
    }
}

macro_rules! bundle_tuple {
    ($( $id:ident ),*) => {

    #[allow(non_snake_case)]
    impl<$($id: Tensors,)*> Tensors for ($($id,)*) {
        fn fun_in(expr: &mut Expr, tensors: &mut TensorCache, input: &Self) -> Self {
            let ($($id,)*) = input;

            (
                $(
                    $id::fun_in(expr, tensors, $id),
                )*
            )
        }

        fn set_input(out: &mut TensorCache, index: usize, item: &Self) -> usize {
            let ($($id,)*) = item;
    
            $(
                let index = $id::set_input(out, index, $id);
            )*
    
            index
        }
                
        fn out_ids(out: &mut Vec<TensorId>, item: &Self) {
            let ($($id,)*) = item;

            $(
                $id::out_ids(out, $id);
            )*
        }

        fn make_out(cache: &TensorCache, ids: &Vec<TensorId>, index: &mut usize) -> Self {
            (
                $(
                    $id::make_out(cache, ids, index),
                )*
            )
        }
    }
}
}

bundle_tuple!(P1, P2);
bundle_tuple!(P1, P2, P3);
bundle_tuple!(P1, P2, P3, P4);
bundle_tuple!(P1, P2, P3, P4, P5);
bundle_tuple!(P1, P2, P3, P4, P5, P6);
bundle_tuple!(P1, P2, P3, P4, P5, P6, P7);
bundle_tuple!(P1, P2, P3, P4, P5, P6, P7, P8);
