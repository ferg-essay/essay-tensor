use crate::tensor::{Type, IntoTensorList, Tensor};

use super::concatenate_axis;

pub fn hstack<D>(x: impl IntoTensorList<D>) -> Tensor<D>
where
    D: Type + Clone
{
    let mut vec = Vec::<Tensor<D>>::new();

    x.into_list(&mut vec);

    let shape = vec[0].shape();

    if shape.rank() == 1 {
        concatenate_axis(0, vec)
    } else {
        concatenate_axis(1, vec)
    }
}

#[cfg(test)]
mod test {
    use crate::{prelude::*, array::hstack};
    
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
