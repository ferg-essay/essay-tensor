use crate::{ten, tensor::{scalar, Shape, Tensor}, test::T};


#[test]
fn tensor_0_from_scalar() {
    let t0 : Tensor = 0.25.into();

    assert_eq!(t0.size(), 1);
    assert_eq!(t0.rank(), 0);
    assert_eq!(t0.shape(), &Shape::scalar());
    assert_eq!(t0.get(0), Some(&0.25));
}

#[test]
fn tensor_empty() {
    let empty: [i32; 0] = [];
    let t = Tensor::from(empty);
    assert_eq!(t.size(), 0);
    assert_eq!(t.rank(), 1);
    assert_eq!(t.cols(), 0);
    assert_eq!(t.as_slice(), &[]);
    assert_eq!(t.shape().as_vec(), &[0]);
}

#[test]
fn tensor_from_f32() {
    let t = Tensor::from(10.5);
    assert_eq!(t.size(), 1);
    assert_eq!(t[0], 10.5);
    assert_eq!(t.as_slice(), &[10.5]);
    assert_eq!(t.shape(), &Shape::scalar());
}

#[test]
fn tensor_scalar_type() {
    let t = Tensor::from(T(10));
    assert_eq!(t.size(), 1);
    assert_eq!(t[0], T(10));
    assert_eq!(t.as_slice(), &[T(10)]);
    assert_eq!(t.shape(), &Shape::scalar());
}

#[test]
fn tensor_from_scalar_iterator() {
    let t0 = Tensor::from_iter(0..6);
    assert_eq!(t0.size(), 6);
    assert_eq!(t0.shape().as_vec(), &[6]);
    for i in 0..6 {
        assert_eq!(t0.get(i), Some(&i));
    }
}

//
// from vec 
//

#[test]
fn tensor_from_vec() {
    let t0 = Tensor::from(vec![10, 11, 12]);
    assert_eq!(t0.size(), 3);
    assert_eq!(t0.rank(), 1);
    assert_eq!(t0.cols(), 3);
    assert_eq!(t0.shape().as_vec(), &[3]);
    assert_eq!(t0.as_slice(), &[10, 11, 12]);
}

#[test]
fn tensor_from_vec_type() {
    let t0 = Tensor::from(vec![T(10), T(11), T(12)]);
    assert_eq!(t0.size(), 3);
    assert_eq!(t0.rank(), 1);
    assert_eq!(t0.cols(), 3);
    assert_eq!(t0.shape().as_vec(), &[3]);
    assert_eq!(t0.as_slice(), &[T(10), T(11), T(12)]);
}

#[test]
fn tensor_from_vec_ref() {
    let t0 = Tensor::from(&vec![10, 11, 12]);
    assert_eq!(t0.size(), 3);
    assert_eq!(t0.shape().as_vec(), &[3]);
    assert_eq!(t0.as_slice(), &[10, 11, 12]);
}

#[test]
fn tensor_from_vec_rows() {
    let t0 = Tensor::from(vec![[10, 20, 30], [11, 21, 31]]);

    assert_eq!(t0.size(), 6);
    assert_eq!(t0.cols(), 3);
    assert_eq!(t0.rows(), 2);
    assert_eq!(t0.shape().as_vec(), &[2, 3]);
    assert_eq!(t0.as_slice(), &[10, 20, 30, 11, 21, 31]);
}

#[test]
fn tensor_from_vec_rows_ref() {
    let t0 = Tensor::from(&vec![[10, 20, 30], [11, 21, 31]]);

    assert_eq!(t0.size(), 6);
    assert_eq!(t0.cols(), 3);
    assert_eq!(t0.rows(), 2);
    assert_eq!(t0.shape().as_vec(), &[2, 3]);
    assert_eq!(t0.as_slice(), &[10, 20, 30, 11, 21, 31]);
}

#[test]
fn tensor_into() {
    assert_eq!(into([3.]), ten![3.]);
}

fn _as_ref(tensor: impl AsRef<Tensor>) -> Tensor {
    tensor.as_ref().clone()
}

fn into(tensor: impl Into<Tensor>) -> Tensor {
    tensor.into()
}


//
// slices (arrays)
//

// Array: [Dtype]
#[test]
fn tensor_from_array_1d() {
    let t0 = Tensor::from([10, 11, 12]);
    assert_eq!(t0.size(), 3);
    assert_eq!(t0.shape().as_vec(), &[3]);
    assert_eq!(t0.as_slice(), &[10, 11, 12]);
}

// Array: &[Dtype]
#[test]
fn tensor_from_array_1d_ref() {
    let t0 = Tensor::from(vec![10, 11, 12].as_slice());
    assert_eq!(t0.size(), 3);
    assert_eq!(t0.shape().as_vec(), &[3]);
    assert_eq!(t0.as_slice(), &[10, 11, 12]);
}

// Array: [[Dtype; N]]
#[test]
fn tensor_from_array_2d() {
    let t0 = Tensor::from([[10, 11], [110, 111], [210, 211]]);
    assert_eq!(t0.size(), 6);
    assert_eq!(t0.shape().as_vec(), &[3, 2]);
    assert_eq!(t0.as_slice(), &[10, 11, 110, 111, 210, 211]);
}

// Array: &[[Dtype; N]]
#[test]
fn tensor_from_array_2d_ref() {
    let vec = vec![
        [10, 11], [110, 111], [210, 211]
    ];

    let t0 = Tensor::from(vec.as_slice());

    assert_eq!(t0.size(), 6);
    assert_eq!(t0.shape().as_vec(), &[3, 2]);
    assert_eq!(t0.as_slice(), &[10, 11, 110, 111, 210, 211]);
}

//
// concatenating tensors
//

#[test]
fn tensor_from_tensor_slice() {
    let t0 = Tensor::from([scalar(2.), scalar(1.), scalar(3.)]);
    assert_eq!(t0.size(), 3);
    assert_eq!(t0.shape().as_vec(), &[3]);
    assert_eq!(t0.get(0), Some(&2.));
    assert_eq!(t0.get(1), Some(&1.));
    assert_eq!(t0.get(2), Some(&3.));

    let t1 = Tensor::from([
        ten![1., 2.], 
        ten![2., 3.], 
        ten![3., 4.]]
    );
    assert_eq!(t1.size(), 6);
    assert_eq!(t1.shape().as_vec(), &[3, 2]);
    assert_eq!(t1[0], 1.);
    assert_eq!(t1[1], 2.);
    assert_eq!(t1[2], 2.);
    assert_eq!(t1[3], 3.);
    assert_eq!(t1[4], 3.);
    assert_eq!(t1[5], 4.);
}

#[test]
fn tensor_from_vec_slice() {
    let vec = vec![ten!(2.), ten!(1.), ten!(3.)];

    let t0 = Tensor::from(vec.as_slice());
    assert_eq!(t0.size(), 3);
    assert_eq!(t0.shape().as_vec(), &[3]);
    assert_eq!(t0.get(0), Some(&2.));
    assert_eq!(t0.get(1), Some(&1.));
    assert_eq!(t0.get(2), Some(&3.));

    let vec = vec![
        ten!([1., 2.]), 
        ten!([2., 3.]), 
        ten!([3., 4.])
    ];

    let ptr = vec.as_slice();
    let t1 = Tensor::from(ptr);

    assert_eq!(t1.size(), 6);
    assert_eq!(t1.shape().as_vec(), &[2, 3]);
    assert_eq!(t1[0], 1.);
    assert_eq!(t1[1], 2.);
    assert_eq!(t1[2], 2.);
    assert_eq!(t1[3], 3.);
    assert_eq!(t1[4], 3.);
    assert_eq!(t1[5], 4.);
    
    let t1 = Tensor::from(&vec);

    assert_eq!(t1.size(), 6);
    assert_eq!(t1.shape().as_vec(), &[2, 3]);
    assert_eq!(t1[0], 1.);
    assert_eq!(t1[1], 2.);
    assert_eq!(t1[2], 2.);
    assert_eq!(t1[3], 3.);
    assert_eq!(t1[4], 3.);
    assert_eq!(t1[5], 4.);
}

#[test]
fn shape_from_zeros() {
    let t = Tensor::<f32>::zeros([3, 2, 4, 5]);
    assert_eq!(t.shape().as_vec(), &[3, 2, 4, 5]);
    assert_eq!(t.rank(), 4);
    assert_eq!(t.cols(), 5);
    assert_eq!(t.rows(), 4);
    assert_eq!(t.size(), 3 * 2 * 4 * 5);
}

#[test]
fn tensor_macro_float() {
    let t = ten![1.];
    assert_eq!(t.shape().as_vec(), [1]);
    assert_eq!(t, Tensor::from([1.]));

    let t = ten![1., 2., 3.];
    assert_eq!(t.shape().as_vec(), [3]);
    assert_eq!(t, Tensor::from([1., 2., 3.]));
    assert_eq!(t, [1., 2., 3.].into());

    let t = ten![[1., 2., 3.], [4., 5., 6.]];
    assert_eq!(t.shape().as_vec(), [2, 3]);
    assert_eq!(t, [[1., 2., 3.], [4., 5., 6.]].into());
}

#[test]
fn tensor_macro_string() {
    let t = ten!("test");
    assert_eq!(t.shape().as_vec(), &[1]);

    assert_eq!(&t[0], "test");

    let t = ten!["t1", "t2", "t3"];
    assert_eq!(t.shape().as_vec(), &[3]);

    assert_eq!(&t[0], "t1");
    assert_eq!(&t[1], "t2");
    assert_eq!(&t[2], "t3");
}
