
//
// reshape operation
//

use crate::tensor::{Shape, Tensor};

pub fn reshape<T>(x: &Tensor<T>, shape: impl Into<Shape>) -> Tensor<T> {
    x.clone().reshape(shape)
}

#[cfg(test)]
mod test {
    use crate::{prelude::*, array::reshape};
    
    #[test]
    fn test_reshape() {
        assert_eq!(
            reshape(&tf32!([[1., 2.], [3., 4.]]), [4]), 
            tf32!([1., 2., 3., 4.])
        );
    }
}
