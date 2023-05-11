use crate::{
    ops::{Uop}
};


#[cfg(test)]
mod test {
    use crate::prelude::{*};

    #[test]
    fn neg_f() {
        assert_eq!(- tensor!(1.), tensor!(-1.));
        assert_eq!(- tensor!([1.]), tensor!([-1.]));
        assert_eq!(- tensor!([[1.]]), tensor!([[-1.]]));
        assert_eq!(- tensor!([[[1.]]]), tensor!([[[-1.]]]));
        assert_eq!(- tensor!([[[[1.]]]]), tensor!([[[[-1.]]]]));

        assert_eq!(- tensor!([1., 2.]), tensor!([-1., -2.]));
        assert_eq!(- tensor!([[-1., -2.], [-3., -4.]]), 
            tensor!([[1., 2.], [3., 4.]]));
    }
}