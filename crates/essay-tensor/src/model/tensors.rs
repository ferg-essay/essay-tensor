use crate::{Tensor, tensor::{TensorId}, model::TensorCache};

pub trait Tensors : Clone {
    type In<'a>;
    type Out;

    fn push_arg(tensors: &mut TensorCache, index: usize, item: &Self) -> usize;
    fn set_arg(tensors: &mut TensorCache, index: usize, item: &Self) -> usize;
    fn make_arg<'a>(tensors: &'a TensorCache, index: &mut usize) -> Self::In<'a>;

    fn out_ids(out: &mut Vec<TensorId>, item: &Self);
    fn make_out(cache: &TensorCache, out: &Vec<TensorId>, index: &mut usize) -> Self::Out;
}

impl Tensors for Tensor {
    type In<'a> = &'a Tensor;
    type Out = Tensor;

    fn push_arg(out: &mut TensorCache, index: usize, item: &Self::Out) -> usize {
        let id = TensorId(index);

        out.push(Some(item.clone().with_id(id)));

        index + 1
    }

    fn set_arg(out: &mut TensorCache, index: usize, item: &Self::Out) -> usize {
        let id = TensorId(index);
        
        out.set(id, item.clone().with_id(id));

        index + 1
    }

    fn make_arg<'a>(cache: &'a TensorCache, index: &mut usize) -> Self::In<'a> {
        let id = TensorId(*index);
        *index += 1;

        // let tensor = cache.get(id).unwrap().clone();
        // tensor

        cache.get(id).unwrap()
    }

    fn out_ids(out: &mut Vec<TensorId>, item: &Self::Out) {
        if item.id().is_some() {
            out.push(item.id())
        }
    }

    fn make_out(cache: &TensorCache, ids: &Vec<TensorId>, index: &mut usize) -> Self::Out {
        let value = cache.get(ids[*index]).unwrap();
        *index += 1;
        value.clone()
    }
}

impl Tensors for &Tensor {
    type In<'a> = &'a Tensor;
    type Out = Tensor;

    //fn push_arg(out: &mut TensorCache, index: usize, item: &Self::Item) -> usize {
    fn push_arg(out: &mut TensorCache, index: usize, item: &Self) -> usize {
        let item = *item;
        let id = TensorId(index);

        out.push(Some(item.clone().with_id(id)));

        index + 1
    }

    //fn set_arg(out: &mut TensorCache, index: usize, item: &Self::Item) -> usize {
    fn set_arg(out: &mut TensorCache, index: usize, item: &Self) -> usize {
        let item = *item;
        let id = TensorId(index);
        
        out.set(id, item.clone().with_id(id));

        index + 1
    }

    fn make_arg<'a>(cache: &'a TensorCache, index: &mut usize) -> Self::In<'a> {
        let id = TensorId(*index);
        *index += 1;

        cache.get(id).unwrap()
    }

    //fn out_ids(out: &mut Vec<TensorId>, item: &Self::Item) {
    fn out_ids(out: &mut Vec<TensorId>, item: &Self) {
        if item.id().is_some() {
            out.push(item.id())
        }
    }

    fn make_out(cache: &TensorCache, ids: &Vec<TensorId>, index: &mut usize) -> Self::Out {
        let value = cache.get(ids[*index]).unwrap();
        *index += 1;
        value.clone()
    }
}

impl Tensors for () {
    type In<'a> = ();
    type Out = ();

    fn push_arg(_out: &mut TensorCache, index: usize, _item: &Self::Out) -> usize {
        index
    }

    fn set_arg(_out: &mut TensorCache, index: usize, _item: &Self::Out) -> usize {
        index
    }

    fn make_arg<'a>(_cache: &'a TensorCache, _index: &mut usize) -> Self::In<'a> {
        ()
    }

    fn out_ids(_out: &mut Vec<TensorId>, _item: &Self::Out) {
    }

    fn make_out(_cache: &TensorCache, _ids: &Vec<TensorId>, _index: &mut usize) -> Self::Out {
        ()
    }
}

macro_rules! bundle_tuple {
    ($( $id:ident ),*) => {

    #[allow(non_snake_case)]
    impl<$($id: Tensors,)*> Tensors for ($($id,)*) {
        type In<'a> = ($($id::In<'a>,)*);
        type Out = ($($id::Out,)*);

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

        fn make_arg<'a>(cache: &'a TensorCache, index: &mut usize) -> Self::In<'a> {
            (
                $(
                    $id::make_arg(cache, index),
                )*
            )
        }

        //fn out_ids(out: &mut Vec<TensorId>, item: &Self::Item) {
        fn out_ids(out: &mut Vec<TensorId>, item: &Self) {
            let ($($id,)*) = item;

            $(
                $id::out_ids(out, $id);
            )*
        }

        fn make_out(cache: &TensorCache, ids: &Vec<TensorId>, index: &mut usize) -> Self::Out {
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
