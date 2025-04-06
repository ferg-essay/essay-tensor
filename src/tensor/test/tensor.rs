use crate::{ten, test::{Dead, Messages, T}};

/// Tensors can be cloned even if their type is non-cloneable because
/// the data isn't cloned
#[test]
fn clone_with_non_clone_type() {
    let ten = ten![T(2), T(3), T(4)];

    assert_eq!(ten.clone(), ten![T(2), T(3), T(4)]);
}


/// Tensor::clone does not clone the data
#[test]
fn clone_drop() {
    Messages::clear();

    {
        let ten = ten![Dead(0x10), Dead(0x20)];
        vec![ten.clone(), ten.clone(), ten.clone()];
    }

    assert_eq!(Messages::take(), vec!["Dead(10)", "Dead(20)"]);

    {
        // check that compiler isn't eliminating clones or drops
        let vec = vec![Dead(0x10), Dead(0x20)];
        vec![vec.clone(), vec.clone(), vec.clone()];
    }

    assert_eq!(Messages::take(), vec!["Dead(10)", "Dead(20)", "Dead(10)", "Dead(20)", "Dead(10)", "Dead(20)",  "Dead(10)", "Dead(20)"]);
}
