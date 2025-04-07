use crate::{ten, tensor::Shape};


#[test]
fn shape_from_slice() {
    let shape = Shape::from([]);
    assert_eq!(shape.rank(), 1);
    assert_eq!(shape.size(), 0);
    assert_eq!(shape.cols(), 0);
    assert_eq!(shape.rows(), 0);
    assert_eq!(shape.as_vec(), vec![0]);

    let shape = Shape::from([4]);
    assert_eq!(shape.rank(), 1);
    assert_eq!(shape.size(), 4);
    assert_eq!(shape.cols(), 4);
    assert_eq!(shape.rows(), 0);
    assert_eq!(shape.dim(0), 4);
    assert_eq!(shape.rdim(0), 4);
    assert_eq!(shape.as_vec(), vec![4]);

    let shape = Shape::from([2, 4]);
    assert_eq!(shape.rank(), 2);
    assert_eq!(shape.size(), 8);
    assert_eq!(shape.cols(), 4);
    assert_eq!(shape.rows(), 2);
    assert_eq!(shape.dim(0), 2);
    assert_eq!(shape.dim(1), 4);
    assert_eq!(shape.rdim(0), 4);
    assert_eq!(shape.rdim(1), 2);
    assert_eq!(shape.as_vec(), vec![2, 4]);

}
#[test]
fn shape_debug() {
    assert_eq!(format!("{:?}", Shape::from([])), "Shape { dims: [0, 0, 0, 0, 0, 0], rank: 0 }");
    assert_eq!(format!("{:?}", Shape::from([4])), "Shape { dims: [4, 0, 0, 0, 0, 0], rank: 1 }");
    assert_eq!(format!("{:?}", Shape::from([4, 2])), "Shape { dims: [2, 4, 0, 0, 0, 0], rank: 2 }");
}

#[test]
fn shape_insert() {
    assert_eq!(Shape::from([1]).insert(0, 4).as_vec(), vec![4, 1]);
    assert_eq!(Shape::from([1]).insert(1, 4).as_vec(), vec![1, 4]);

    assert_eq!(Shape::from([1, 2]).insert(0, 4).as_vec(), vec![4, 1, 2]);
    assert_eq!(Shape::from([1, 2]).insert(1, 4).as_vec(), vec![1, 4, 2]);
    assert_eq!(Shape::from([1, 2]).insert(2, 4).as_vec(), vec![1, 2, 4]);
}

#[test]
fn shape_rinsert() {
    assert_eq!(Shape::from([1]).rinsert(0, 4).as_vec(), vec![1, 4]);
    assert_eq!(Shape::from([1]).rinsert(1, 4).as_vec(), vec![4, 1]);

    assert_eq!(Shape::from([1, 2]).rinsert(0, 4).as_vec(), vec![1, 2, 4]);
    assert_eq!(Shape::from([1, 2]).rinsert(1, 4).as_vec(), vec![1, 4, 2]);
    assert_eq!(Shape::from([1, 2]).rinsert(2, 4).as_vec(), vec![4, 1, 2]);
}

#[test]
fn shape_replace() {
    assert_eq!(Shape::from([1]).replace(0, 4).as_vec(), vec![4]);
    assert_eq!(Shape::from([1, 2, 3]).replace(0, 4).as_vec(), vec![4, 2, 3]);
    assert_eq!(Shape::from([1, 2, 3]).replace(1, 4).as_vec(), vec![1, 4, 3]);
    assert_eq!(Shape::from([1, 2, 3]).replace(2, 4).as_vec(), vec![1, 2, 4]);
}

#[test]
fn shape_rreplace() {
    assert_eq!(Shape::from([1]).rreplace(0, 4).as_vec(), vec![4]);
    assert_eq!(Shape::from([1, 2, 3]).rreplace(0, 4).as_vec(), vec![1, 2, 4]);
    assert_eq!(Shape::from([1, 2, 3]).rreplace(1, 4).as_vec(), vec![1, 4, 3]);
    assert_eq!(Shape::from([1, 2, 3]).rreplace(2, 4).as_vec(), vec![4, 2, 3]);
}

#[test]
fn shape_remove() {
    assert_eq!(Shape::from([1]).remove(0).as_vec(), vec![]);
    assert_eq!(Shape::from([1, 2, 3]).remove(0).as_vec(), vec![2, 3]);
    assert_eq!(Shape::from([1, 2, 3]).remove(1).as_vec(), vec![1, 3]);
    assert_eq!(Shape::from([1, 2, 3]).remove(2).as_vec(), vec![1, 2]);
}

#[test]
fn shape_rremove() {
    assert_eq!(Shape::from([1]).rremove(0).as_vec(), vec![]);
    assert_eq!(Shape::from([1, 2, 3]).rremove(0).as_vec(), vec![1, 2]);
    assert_eq!(Shape::from([1, 2, 3]).rremove(1).as_vec(), vec![1, 3]);
    assert_eq!(Shape::from([1, 2, 3]).rremove(2).as_vec(), vec![2, 3]);
}

#[test]
fn test_flatten() {
    assert_eq!(ten!([[1., 2.], [3., 4.]]).flatten(), ten!([1., 2., 3., 4.]));
}
#[test]
fn test_reshape() {
    assert_eq!(
        ten!([[1., 2.], [3., 4.]]).reshape([4]),
        ten!([1., 2., 3., 4.])
    );
}

#[test]
fn test_squeeze() {
    assert_eq!(ten!([[1.]]).squeeze(), ten!(1.));
    assert_eq!(ten!([[1., 2.]]).squeeze(), ten!([1., 2.]));
    assert_eq!(ten!([[[1.], [2.]]]).squeeze(), ten!([1., 2.]));
}

#[test]
fn test_squeeze_axis() {
    assert_eq!(ten!([[[1.], [2.]]]).squeeze_axis(-1), ten!([[1., 2.]]));
}
