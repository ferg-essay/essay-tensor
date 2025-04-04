
//
// reshape operation
//

use crate::tensor::{Shape, Tensor, Type};

pub fn reshape<T: Type>(x: &Tensor<T>, shape: impl Into<Shape>) -> Tensor<T> {
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
