use crate::{Tensor, tensor::{TensorId}, model::TensorCache, prelude::Shape};

use super::{ModelIn, model::ModelInner};

pub trait Tensors : Clone {
    type Item;
    type Shape;
    type ModelIn;

    fn push_arg(tensors: &mut TensorCache, index: usize, item: &Self) -> usize;
    fn set_arg(tensors: &mut TensorCache, index: usize, item: &Self) -> usize;
    fn make_arg(tensors: &TensorCache, index: &mut usize) -> Self::Item;

    fn out_ids(out: &mut Vec<TensorId>, item: &Self);
    fn make_out(cache: &TensorCache, out: &Vec<TensorId>, index: &mut usize) -> Self::Item;

    fn model_in(builder: &ModelInner, input: &Self) -> Self::ModelIn;
    fn model_out(out: &mut Vec<TensorId>, input: &Self::ModelIn);
}

impl Tensors for Tensor<f32> {
    type Item = Tensor<f32>;
    type Shape = Shape;
    type ModelIn = ModelIn<f32>;

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

    fn model_in(builder: &ModelInner, input: &Self) -> Self::ModelIn {
        builder.arg(input)
    }

    fn model_out(out: &mut Vec<TensorId>, item: &Self::ModelIn) {
        assert!(item.id().is_some());

        out.push(item.id());
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

impl Tensors for () {
    type Item = ();
    type Shape = ();
    type ModelIn = ();

    fn push_arg(_out: &mut TensorCache, index: usize, _item: &Self::Item) -> usize {
        index
    }

    fn set_arg(_out: &mut TensorCache, index: usize, _item: &Self::Item) -> usize {
        index
    }

    fn make_arg<'a>(_cache: &'a TensorCache, _index: &mut usize) -> Self::Item {
        ()
    }

    fn model_in(builder: &ModelInner, input: &Self) -> Self::ModelIn {
        ()
    }

    fn model_out(out: &mut Vec<TensorId>, item: &Self::ModelIn) {
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
    type ModelIn = Vec<T::ModelIn>;

    fn push_arg(_out: &mut TensorCache, _index: usize, _item: &Self) -> usize {
        todo!()
    }

    fn set_arg(_out: &mut TensorCache, _index: usize, _item: &Self) -> usize {
        todo!()
    }

    fn make_arg<'a>(_cache: &'a TensorCache, _index: &mut usize) -> Self::Item {
        todo!()
    }

    fn model_in(builder: &ModelInner, input: &Self) -> Self::ModelIn {
        todo!()
    }

    fn model_out(out: &mut Vec<TensorId>, item: &Self::ModelIn) {
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
        type ModelIn = ($($id::ModelIn,)*);

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

        fn model_in(builder: &ModelInner, input: &Self) -> Self::ModelIn {
            let ($($id,)*) = input;

            (
                $(
                    $id::model_in(builder, $id),
                )*
            )
        }

        fn model_out(out: &mut Vec<TensorId>, item: &Self::ModelIn) {
            let ($($id,)*) = item;
    
            $(
                $id::model_out(out, $id);
            )*
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
