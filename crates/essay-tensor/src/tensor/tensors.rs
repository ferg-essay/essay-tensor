use crate::{Tensor, tensor::{TensorId}, function::TensorCache};

pub trait Tensors : Clone {
    type Item;

    fn push_arg(tensors: &mut TensorCache, index: usize, item: &Self::Item) -> usize;
    fn set_arg(tensors: &mut TensorCache, index: usize, item: &Self::Item) -> usize;
    fn make_arg(tensors: &TensorCache, index: &mut usize) -> Self::Item;

    fn out_ids(out: &mut Vec<TensorId>, item: &Self::Item);
    fn make_out(cache: &TensorCache, out: &Vec<TensorId>, index: &mut usize) -> Self::Item;
}

impl Tensors for Tensor {
    type Item = Tensor;

    fn push_arg(out: &mut TensorCache, index: usize, item: &Self::Item) -> usize {
        let id = TensorId(index);

        out.push(Some(item.clone().with_id(id)));

        index + 1
    }

    fn set_arg(out: &mut TensorCache, index: usize, item: &Self::Item) -> usize {
        let id = TensorId(index);
        
        out.set(id, item.clone().with_id(id));

        index + 1
    }

    fn make_arg(cache: &TensorCache, index: &mut usize) -> Self::Item {
        let id = TensorId(*index);
        *index += 1;

        let tensor = cache.get(id).unwrap().clone();

        tensor
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

impl Tensors for () {
    type Item = ();

    fn push_arg(_out: &mut TensorCache, index: usize, _item: &Self::Item) -> usize {
        index
    }

    fn set_arg(_out: &mut TensorCache, index: usize, _item: &Self::Item) -> usize {
        index
    }

    fn make_arg(_cache: &TensorCache, _index: &mut usize) -> Self::Item {
        ()
    }

    fn out_ids(_out: &mut Vec<TensorId>, _item: &Self::Item) {
    }

    fn make_out(_cache: &TensorCache, _ids: &Vec<TensorId>, _index: &mut usize) -> Self::Item {
        ()
    }
}

macro_rules! bundle_tuple {
    ($( $id:ident ),*) => {

    #[allow(non_snake_case)]
    impl<$($id: Tensors<Item=$id>,)*> Tensors for ($($id,)*) {
        type Item = ($($id,)*);

        fn push_arg(out: &mut TensorCache, index: usize, item: &Self::Item) -> usize {
            let ($($id,)*) = item;

            $(
                let index = $id::push_arg(out, index, $id);
            )*

            index
        }

        fn set_arg(out: &mut TensorCache, index: usize, item: &Self::Item) -> usize {
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

        fn out_ids(out: &mut Vec<TensorId>, item: &Self::Item) {
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
