use crate::{Tensor, tensor::{TensorId}, model::TensorCache, prelude::Shape};

use super::{Expr};

pub trait Tensors : Clone {
    type Item;
    type Shape;

    /*
    fn push_arg(tensors: &mut TensorCache, index: usize, item: &Self) -> usize;
    fn make_arg(tensors: &TensorCache, index: &mut usize) -> Self::Item;
    */

    fn fun_in(expr: &mut Expr, tensors: &mut TensorCache, input: &Self) -> Self;
    fn set_input(tensors: &mut TensorCache, index: usize, item: &Self) -> usize;

    fn out_ids(out: &mut Vec<TensorId>, item: &Self);
    fn make_out(cache: &TensorCache, out: &Vec<TensorId>, index: &mut usize) -> Self::Item;
}

impl Tensors for Tensor<f32> {
    type Item = Tensor<f32>;
    type Shape = Shape;

    /*
    fn push_arg(out: &mut TensorCache, index: usize, item: &Self::Item) -> usize {
        let id = out.new_id(index);

        out.push(Some(item.clone().with_id(id)));

        index + 1
    }

    fn set_arg(out: &mut TensorCache, index: usize, item: &Self::Item) -> usize {
        let id = out.new_id(index);
        
        out.set(id, item.clone().with_id(id));

        index + 1
    }

    fn make_arg<'a>(out: &'a TensorCache, index: &mut usize) -> Self::Item {
        let id = out.new_id(*index);
        *index += 1;

        // let tensor = cache.get(id).unwrap().clone();
        // tensor

        out.get(id).unwrap().clone()
    }
    */

    fn fun_in(expr: &mut Expr, tensors: &mut TensorCache, input: &Self) -> Self {
        let tensor = expr.arg(input.clone());

        let index = tensors.push(Some(tensor.clone()));
        assert_eq!(tensor.id().index(), index);

        tensor
    }

    fn set_input(out: &mut TensorCache, index: usize, item: &Self::Item) -> usize {
        let id = out.new_id(index);
        
        out.set(id, item.clone().with_id(id));

        index + 1
    }


    fn out_ids(out: &mut Vec<TensorId>, item: &Self::Item) {
        if item.id().is_some() {
            out.push(item.id())
        }
    }

    fn make_out(cache: &TensorCache, ids: &Vec<TensorId>, index: &mut usize) -> Self::Item {
        let value = cache.get(ids[*index]).unwrap();
        *index += 1;
        value.clone()
    }
}
/*
impl Tensors for &Tensor {
    type Item = Tensor;
    type Shape = Shape;
    type ModelIn = ModelIn<f32>;

    //fn push_arg(out: &mut TensorCache, index: usize, item: &Self::Item) -> usize {
    fn push_arg(out: &mut TensorCache, index: usize, item: &Self) -> usize {
        let item = *item;
        let id = out.new_id(index);

        out.push(Some(item.clone().with_id(id)));

        index + 1
    }

    //fn set_arg(out: &mut TensorCache, index: usize, item: &Self::Item) -> usize {
    fn set_arg(out: &mut TensorCache, index: usize, item: &Self) -> usize {
        let item = *item;
        let id = out.new_id(index);
        
        out.set(id, item.clone().with_id(id));

        index + 1
    }

    fn make_arg(out: &TensorCache, index: &mut usize) -> Self::Item {
        let id = out.new_id(*index);
        *index += 1;

        out.get(id).unwrap().clone()
    }

    fn model_in(builder: &ModelInner, input: &Self) -> Self::ModelIn {
        builder.arg(input)
    }

    fn model_out(out: &mut Vec<TensorId>, item: &Self::ModelIn) {
        assert!(item.id().is_some());

        out.push(item.id());
    }

    //fn out_ids(out: &mut Vec<TensorId>, item: &Self::Item) {
    fn out_ids(out: &mut Vec<TensorId>, item: &Self) {
        if item.id().is_some() {
            out.push(item.id())
        }
    }

    fn make_out(cache: &TensorCache, ids: &Vec<TensorId>, index: &mut usize) -> Self::Item {
        let value = cache.get(ids[*index]).unwrap();
        *index += 1;
        value.clone()
    }
}
*/
impl Tensors for () {
    type Item = ();
    type Shape = ();

    /*
    fn push_arg(_out: &mut TensorCache, index: usize, _item: &Self::Item) -> usize {
        index
    }

    fn set_arg(_out: &mut TensorCache, index: usize, _item: &Self::Item) -> usize {
        index
    }

    fn make_arg<'a>(_cache: &'a TensorCache, _index: &mut usize) -> Self::Item {
        ()
    }
    */

    fn fun_in(expr: &mut Expr, tensors: &mut TensorCache, input: &Self) -> Self {
        ()
    }

    fn set_input(_out: &mut TensorCache, index: usize, _item: &Self::Item) -> usize {
        index
    }

    fn out_ids(_out: &mut Vec<TensorId>, _item: &Self::Item) {
    }

    fn make_out(_cache: &TensorCache, _ids: &Vec<TensorId>, _index: &mut usize) -> Self::Item {
        ()
    }
}

impl<T: Tensors> Tensors for Vec<T> {
    type Item = Vec<T::Item>;
    type Shape = Vec<T::Shape>;

    /*
    fn push_arg(_out: &mut TensorCache, _index: usize, _item: &Self) -> usize {
        todo!()
    }

    fn set_arg(_out: &mut TensorCache, _index: usize, _item: &Self) -> usize {
        todo!()
    }

    fn make_arg<'a>(_cache: &'a TensorCache, _index: &mut usize) -> Self::Item {
        todo!()
    }
    */

    fn fun_in(expr: &mut Expr, tensors: &mut TensorCache, input: &Self) -> Self {
        todo!()
    }

    fn set_input(_out: &mut TensorCache, _index: usize, _item: &Self) -> usize {
        todo!()
    }

    fn out_ids(_out: &mut Vec<TensorId>, _item: &Self) {
        todo!()
    }

    fn make_out(_cache: &TensorCache, _ids: &Vec<TensorId>, _index: &mut usize) -> Self::Item {
        todo!()
    }
}

macro_rules! bundle_tuple {
    ($( $id:ident ),*) => {

    #[allow(non_snake_case)]
    impl<$($id: Tensors,)*> Tensors for ($($id,)*) {
        type Item = ($($id::Item,)*);
        type Shape = ($($id::Shape,)*);

        /*
        //fn push_arg(out: &mut TensorCache, index: usize, item: &Self::Item) -> usize {
        fn push_arg(out: &mut TensorCache, index: usize, item: &Self) -> usize {
            let ($($id,)*) = item;

            $(
                let index = $id::push_arg(out, index, $id);
            )*

            index
        }

        //fn set_arg(out: &mut TensorCache, index: usize, item: &Self::Item) -> usize {
        fn set_arg(out: &mut TensorCache, index: usize, item: &Self) -> usize {
            let ($($id,)*) = item;

            $(
                let index = $id::set_arg(out, index, $id);
            )*

            index
        }

        fn make_arg(cache: &TensorCache, index: &mut usize) -> Self::Item {
            (
                $(
                    $id::make_arg(cache, index),
                )*
            )
        }
        */

        fn fun_in(expr: &mut Expr, tensors: &mut TensorCache, input: &Self) -> Self {
            let ($($id,)*) = input;

            (
                $(
                    $id::fun_in(expr, tensors, $id),
                )*
            )
        }

        //fn set_arg(out: &mut TensorCache, index: usize, item: &Self::Item) -> usize {
        fn set_input(out: &mut TensorCache, index: usize, item: &Self) -> usize {
            let ($($id,)*) = item;
    
            $(
                let index = $id::set_input(out, index, $id);
            )*
    
            index
        }
                
        //fn out_ids(out: &mut Vec<TensorId>, item: &Self::Item) {
        fn out_ids(out: &mut Vec<TensorId>, item: &Self) {
            let ($($id,)*) = item;

            $(
                $id::out_ids(out, $id);
            )*
        }

        fn make_out(cache: &TensorCache, ids: &Vec<TensorId>, index: &mut usize) -> Self::Item {
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
