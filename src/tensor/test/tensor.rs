use crate::{init::fill, ten, tensor::{Shape, Tensor}, test::{Dead, Messages, T}};

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

#[test]
fn basic_tensor() {
    let t: Tensor<i32> = Tensor::from(&vec![
        vec![
            vec![101, 102, 103, 104],
            vec![111, 112, 113, 114],
            vec![121, 122, 123, 124],
        ],
        vec![
            vec![201, 202, 203, 204],
            vec![211, 212, 213, 214],
            vec![221, 222, 223, 224],
        ]
    ]);

    assert_eq!(t.size(), 2 * 3 * 4);
    assert_eq!(t.offset(), 0);
    assert_eq!(t.shape(), &[2, 3, 4].into());
    assert_eq!(t.rank(), 3);
    assert_eq!(t.dim(0), 2);
    assert_eq!(t.dim(1), 3);
    assert_eq!(t.dim(2), 4);
    assert_eq!(t.rdim(0), 4);
    assert_eq!(t.rdim(1), 3);
    assert_eq!(t.rdim(2), 2);
    assert_eq!(t.cols(), 4);
    assert_eq!(t.rows(), 3);

    let slice = [
        101, 102, 103, 104,
        111, 112, 113, 114,
        121, 122, 123, 124,
        201, 202, 203, 204,
        211, 212, 213, 214,
        221, 222, 223, 224,
    ];

    assert_eq!(t.as_slice(), &slice);

    let ten2 = Tensor::from(Vec::from(&slice));
    assert_ne!(t, ten2);
    assert_eq!(t, ten2.reshape([2, 3, 4]));

    for (i, v) in slice.iter().enumerate() {
        assert_eq!(t.get(i).unwrap(), v);
    }

    let vec: Vec<i32> = t.iter().map(|v| *v).collect();
    assert_eq!(vec, Vec::from(&slice));
}

#[test]
fn debug_tensor_from_f32() {
    let t = Tensor::from(10.5);
    assert_eq!(format!("{:?}", t), "Tensor {10.5, shape: [], dtype: f32");
}

#[test]
fn debug_vector_from_slice_f32() {
    let t = Tensor::from([10.5]);
    assert_eq!(format!("{:?}", t), "Tensor{[10.5], shape: [1], dtype: f32}");

    let t = Tensor::from([1., 2.]);
    assert_eq!(format!("{:?}", t), "Tensor{[1 2], shape: [2], dtype: f32}");

    let t = Tensor::from([1., 2., 3., 4., 5.]);
    assert_eq!(format!("{:?}", t), "Tensor{[1 2 3 4 5], shape: [5], dtype: f32}");
}

#[test]
fn debug_matrix_from_slice_f32() {
    let t = Tensor::from([[10.5]]);
    assert_eq!(format!("{:?}", t), "Tensor<f64> {\n[[10.5]], shape: [1, 1]}");

    let t = Tensor::from([[1., 2.]]);
    assert_eq!(format!("{:?}", t), "Tensor{\n[[1 2]], shape: [1, 2], dtype: f32}");

    let t = Tensor::from([[1., 2., 3.], [4., 5., 6.]]);
    assert_eq!(format!("{:?}", t), "Tensor{\n[[1 2 3],\n [4 5 6]], shape: [2, 3], dtype: f32}");
}

#[test]
fn debug_tensor3_from_slice_f32() {
    let t = Tensor::<f32>::from([
        [[10.5]]
    ]);
    assert_eq!(format!("{:?}", t), "Tensor{\n[[[10.5]]], shape: [1, 1, 1], dtype: f32}");

    let t = Tensor::<f32>::from([
        [[1., 2.]],
        [[101., 102.]]
    ]);
    assert_eq!(format!("{:?}", t), "Tensor{\n[[[1 2]],\n\n  [[101 102]]], shape: [2, 1, 2], dtype: f32}");

    let t = Tensor::<f32>::from([
        [[1.0, 2.], [3., 4.]],
        [[101., 102.], [103., 104.]]
    ]);
    assert_eq!(format!("{:?}", t), "Tensor{\n[[[1 2],\n [3 4]],\n\n  [[101 102],\n [103 104]]], shape: [2, 2, 2], dtype: f32}");
}

#[test]
fn debug_vector_from_macro() {
    let t = ten![1.];
    assert_eq!(format!("{:?}", t), "Tensor<f64> {[1.0], shape: [1]}");

    let t = ten![1.];
    assert_eq!(format!("{:?}", t), "Tensor<f64> {[1.0], shape: [1]}");

    let t = ten![1., 2.];
    assert_eq!(format!("{:?}", t), "Tensor<f64> {[1.0 2.0], shape: [2]}");

    let t = ten![[1.0f32, 2., 3.], [3., 4., 5.]];
    assert_eq!(format!("{:?}", t), "Tensor<f32> {\n[[1.0 2.0 3.0],\n [3.0 4.0 5.0]], shape: [2, 3]}");

    let t = ten![
        [[1.0f32, 2.], [3., 4.]],
        [[11., 12.], [13., 14.]]
    ];
    assert_eq!(format!("{:?}", t), "Tensor<f32> {\n[[[1.0 2.0],\n [3.0 4.0]],\n\n  [[11.0 12.0],\n [13.0 14.0]]], shape: [2, 2, 2]}");

    let t = ten![
        [[1., 2.], [3., 4.]],
        [[11., 12.], [13., 14.]],
        [[21., 22.], [23., 24.]]
    ];
    assert_eq!(format!("{:?}", t), "Tensor<f64> {\n[[[1.0 2.0],\n [3.0 4.0]],\n\n  [[11.0 12.0],\n [13.0 14.0]],\n\n  [[21.0 22.0],\n [23.0 24.0]]], shape: [3, 2, 2]}");
}

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

#[test]
fn tensor_init_zero() {
    let t0 = Tensor::init([4], || 0.);

    assert_eq!(t0.size(), 4);
    assert_eq!(t0.cols(), 4);
    assert_eq!(t0.rows(), 0);
    assert_eq!(t0.shape().as_vec(), &[4]);
    assert_eq!(t0, ten!([0., 0., 0., 0.]));

    let t0 = Tensor::init([3, 2], || 0.);

    assert_eq!(t0.size(), 6);
    assert_eq!(t0.cols(), 2);
    assert_eq!(t0.rows(), 3);
    assert_eq!(t0.shape().as_vec(), &[3, 2]);
    assert_eq!(t0, ten!([[0., 0.], [0., 0.], [0., 0.]]));
}

#[test]
fn tensor_init_type() {
    let t0 = Tensor::init([4], || T(4));

    assert_eq!(t0.size(), 4);
    assert_eq!(t0.cols(), 4);
    assert_eq!(t0.rows(), 0);
    assert_eq!(t0.shape().as_vec(), &[4]);
    assert_eq!(t0, ten![T(4), T(4), T(4), T(4)]);
}

#[test]
fn tensor_init_count() {
    let mut count = 0;
    let t0 = Tensor::init([4], || {
        let value = count;
        count += 1;
        value
    });

    assert_eq!(t0.size(), 4);
    assert_eq!(t0.cols(), 4);
    assert_eq!(t0.rows(), 0);
    assert_eq!(t0.shape().as_vec(), &[4]);
    assert_eq!(t0, ten!([0, 1, 2, 3]));

    let mut count = 0;
    let t0 = Tensor::init([3, 2], || {
        let value = count;
        count += 1;
        value
    });

    assert_eq!(t0.size(), 6);
    assert_eq!(t0.cols(), 2);
    assert_eq!(t0.rows(), 3);
    assert_eq!(t0.shape().as_vec(), &[3, 2]);
    assert_eq!(t0, ten!([[0, 1], [2, 3], [4, 5]]));
}

#[test]
fn tensor_init_indexed() {
    let t0 = Tensor::init_rindexed([4], |idx| idx[0]);

    assert_eq!(t0.size(), 4);
    assert_eq!(t0.cols(), 4);
    assert_eq!(t0.rows(), 0);
    assert_eq!(t0.shape().as_vec(), &[4]);
    assert_eq!(t0, ten![0, 1, 2, 3]);

    let t0 = Tensor::init_rindexed([3, 2], |idx| idx[0]);

    assert_eq!(t0, ten![[0, 1], [0, 1], [0, 1]]);
    assert_eq!(t0.size(), 6);
    assert_eq!(t0.cols(), 2);
    assert_eq!(t0.rows(), 3);
    assert_eq!(t0.shape().as_vec(), &[3, 2]);

    let t0 = Tensor::init_rindexed([3, 2], |idx| idx[1]);

    assert_eq!(t0, ten![[0, 0], [1, 1], [2, 2]]);
    assert_eq!(t0.size(), 6);
    assert_eq!(t0.cols(), 2);
    assert_eq!(t0.rows(), 3);
    assert_eq!(t0.shape().as_vec(), &[3, 2]);

    let t0 = Tensor::init_rindexed([3, 2], |idx| 
        if idx[1] == idx[0] { 1 } else { 0 }
    );

    assert_eq!(t0, ten![[1, 0], [0, 1], [0, 0]]);
    assert_eq!(t0.size(), 6);
    assert_eq!(t0.cols(), 2);
    assert_eq!(t0.rows(), 3);
    assert_eq!(t0.shape().as_vec(), &[3, 2]);
}

#[test]
fn tensor_init_indexed_type() {
    let t0 = Tensor::init_rindexed([4], |idx| T(idx[0]));
    assert_eq!(t0, ten![T(0), T(1), T(2), T(3)]);
}

#[test]
fn tensor_fill() {
    let t0 = Tensor::fill([4], 0);

    assert_eq!(t0, ten!([0, 0, 0, 0]));
    assert_eq!(t0.size(), 4);
    assert_eq!(t0.cols(), 4);
    assert_eq!(t0.rows(), 0);
    assert_eq!(t0.shape().as_vec(), &[4]);

    let t0 = fill([3, 2], 0);

    assert_eq!(t0, ten![[0, 0], [0, 0], [0, 0]]);
    assert_eq!(t0.size(), 6);
    assert_eq!(t0.cols(), 2);
    assert_eq!(t0.rows(), 3);
    assert_eq!(t0.shape().as_vec(), &[3, 2]);
}

#[test]
fn tensor_zeros() {
    let t0 = Tensor::zeros([4]);

    assert_eq!(t0.size(), 4);
    assert_eq!(t0.cols(), 4);
    assert_eq!(t0.rows(), 0);
    assert_eq!(t0.shape().as_vec(), &[4]);
    assert_eq!(t0, ten!([0, 0, 0, 0]));

    let t0 = Tensor::zeros([3, 2]);

    assert_eq!(t0.size(), 6);
    assert_eq!(t0.cols(), 2);
    assert_eq!(t0.rows(), 3);
    assert_eq!(t0.shape().as_vec(), &[3, 2]);
    assert_eq!(t0, ten!([[0., 0.], [0., 0.], [0., 0.]]));
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
    let t0 = Tensor::from([ten!(2.), ten!(1.), ten!(3.)]);
    assert_eq!(t0.size(), 3);
    assert_eq!(t0.shape().as_vec(), &[3]);
    assert_eq!(t0.get(0), Some(&2.));
    assert_eq!(t0.get(1), Some(&1.));
    assert_eq!(t0.get(2), Some(&3.));

    let t1 = Tensor::from([
        ten!([1., 2.]), 
        ten!([2., 3.]), 
        ten!([3., 4.])]
    );
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
    assert_eq!(t.shape().batch_len(2), 6);
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

#[test]
fn tensor_iter() {
    let vec : Vec<u32> = ten!([1, 2, 3, 4]).iter().map(|v| *v).collect();
    let vec2 : Vec<u32> = vec!(1, 2, 3, 4);
    assert_eq!(vec, vec2);

    let vec : Vec<u32> = ten!([[1, 2], [3, 4]]).iter().map(|v| *v).collect();
    let vec2 : Vec<u32> = vec!(1, 2, 3, 4);
    assert!(vec.iter().zip(vec2.iter()).all(|(x, y)| x == y));
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

#[test]
fn tensor_iter_slice() {
    let vec : Vec<Vec<u32>> = ten!([1, 2, 3, 4]).iter_row().map(|v| Vec::from(v)).collect();
    let vec2 : Vec<Vec<u32>> = vec!(vec!(1), vec!(2), vec!(3), vec!(4));
    assert_eq!(vec, vec2);

    let vec : Vec<Vec<u32>> = ten!([[1, 2], [3, 4]]).iter_row().map(|v| Vec::from(v)).collect();
    let vec2 : Vec<Vec<u32>> = vec!(vec![1, 2], vec![3, 4]);
    assert_eq!(vec, vec2);
}

#[test]
fn tensor_map_i32() {
    let t1 = ten!([1, 2, 3, 4]);
    let t2 = t1.map(|v| 2 * v);

    assert_eq!(t1.shape(), t2.shape());
    assert_eq!(t1.offset(), t2.offset());
    assert_eq!(t1.size(), t2.size());
    assert_eq!(t2, ten!([2, 4, 6, 8]));
}

#[test]
fn tensor_map_i32_to_f32() {
    let t1 = ten!([1, 2, 3, 4]);
    let t2 = t1.map(|v| 2. * *v as f32);

    assert_eq!(t1.shape(), t2.shape());
    assert_eq!(t1.offset(), t2.offset());
    assert_eq!(t1.size(), t2.size());
    assert_eq!(t2, Tensor::from([2., 4., 6., 8.]));
}

#[test]
fn tensor_fold_i32() {
    let t1 = ten!([1, 2, 3, 4]);
    let t2 = t1.fold(0, |s, v| s + v);

    assert_eq!(t2.shape().as_vec(), &[1]);
    assert_eq!(t2.offset(), 0);
    assert_eq!(t2.size(), 1);
    assert_eq!(t2, ten!([10]));
}

#[test]
fn tensor_reduce() {
    let t1 = ten![1, 2, 3, 4];
    let t2 = t1.reduce(|s, v| s + v);
    assert_eq!(t2, ten![10]);

    assert_eq!(t2.shape().as_vec(), &[1]);
    assert_eq!(t2.offset(), 0);
    assert_eq!(t2.size(), 1);

    let t1 = ten![[1, 2], [3, 4]];
    let t2 = t1.reduce(|s, v| s + v);
    assert_eq!(t2, ten![3, 7]);

    assert_eq!(t2.shape().as_vec(), &[2]);
    assert_eq!(t2.offset(), 0);
    assert_eq!(t2.size(), 2);
}

#[test]
fn test_as_ref() {
    let t1 = ten!([1, 2, 3]);

    as_ref_i32(t1);
}

fn as_ref_i32(tensor: impl AsRef<Tensor<i32>>) {
    let _my_ref = tensor.as_ref();
}

#[test]
fn test_as_slice() {
    let t1 = ten!([1, 2, 3]);

    as_slice(t1);
}

fn as_slice(slice: impl AsRef<[i32]>) {
    let _my_ref = slice.as_ref();
}

#[test]
fn test_collect() {
    let t1: Tensor<i32> = [1, 2, 3].iter().collect();

    assert_eq!(t1, ten!([1, 2, 3]));

    let t1: Tensor<i32> = [1, 2, 3].into_iter().collect();

    assert_eq!(t1, ten!([1, 2, 3]));
}

#[test]
fn test_iter() {
    let mut vec = Vec::new();

    let t1: Tensor<i32> = [1, 2, 3].iter().collect();
    for v in &t1 {
        vec.push(*v);
    }

    assert_eq!(vec, vec![1, 2, 3]);
}
