use crate::{ten, tensor::Tensor, test::{Dead, Messages, T}};

#[test]
fn tensor_rank_0() {
    let t: Tensor<T> = Tensor::from(T(10));

    assert_eq!(t.size(), 1);
    assert_eq!(t.offset(), 0);
    assert_eq!(t.shape().as_vec(), vec![]);
    assert_eq!(t.rank(), 0);

    let slice = [
        T(10)
    ];

    assert_eq!(t.as_slice(), &slice);

    assert_eq!(t, Tensor::from(T(10)));
    assert_ne!(t, Tensor::from(T(11)));
    let v: [T; 0] = [];
    assert_ne!(t, Tensor::<T>::from(v));
    assert_ne!(t, Tensor::from(vec![T(10)]));

    assert_eq!(t, t.clone().reshape([]));
    assert_ne!(t, t.clone().reshape([1]));
    assert_eq!(t.clone().reshape([1]), t.clone().reshape([1]));

    for (i, v) in slice.iter().enumerate() {
        assert_eq!(t.get(i).unwrap(), v);
    }
    assert_eq!(t.get(1), None);

    assert_eq!(t.iter().len(), t.size());
    for (a, b) in t.iter().zip(slice.iter()) {
        assert_eq!(a, b);
    }
}

#[test]
fn tensor_rank_1() {
    let t: Tensor<T> = Tensor::from(vec![T(0), T(1), T(2)]);

    assert_eq!(t.size(), 3);
    assert_eq!(t.offset(), 0);
    assert_eq!(t.shape().as_vec(), vec![3]);
    assert_eq!(t.rank(), 1);
    assert_eq!(t.dim(0), 3);
    assert_eq!(t.rdim(0), 3);
    assert_eq!(t.cols(), 3);

    let slice = [
        T(0), T(1), T(2),
    ];

    assert_eq!(t.as_slice(), &slice);

    assert_eq!(t, Tensor::from(vec![T(0), T(1), T(2)]));
    assert_ne!(t, Tensor::from(vec![T(0), T(1), T(3)]));
    assert_ne!(t, Tensor::from(vec![T(0), T(1)]));
    assert_ne!(t, Tensor::from(vec![T(0), T(1), T(3), T(4)]));

    assert_eq!(t, t.clone().reshape([3]));
    assert_ne!(t, t.clone().reshape([1, 3]));
    assert_ne!(t, t.clone().reshape([3, 1]));
    assert_ne!(t.clone().reshape([1, 3]), t.clone().reshape([3, 1]));

    for (i, v) in slice.iter().enumerate() {
        assert_eq!(t.get(i).unwrap(), v);
    }
    assert_eq!(t.get(4), None);

    assert_eq!(t.iter().len(), t.size());
    for (a, b) in t.iter().zip(slice.iter()) {
        assert_eq!(a, b);
    }
}

#[test]
fn tensor_rank_1_empty() {
    let v: [T; 0] = [];
    let t: Tensor<T> = Tensor::from(v);

    assert_eq!(t.size(), 0);
    assert_eq!(t.offset(), 0);
    assert_eq!(t.shape().as_vec(), vec![0]);
    assert_eq!(t.rank(), 1);
    assert_eq!(t.dim(0), 0);
    assert_eq!(t.rdim(0), 0);
    assert_eq!(t.cols(), 0);

    let slice: [T; 0] = [];

    assert_eq!(t.as_slice(), &slice);

    let vec: Vec<T> = Vec::new();
    assert_eq!(t, Tensor::from(vec));
    assert_ne!(t, Tensor::from(T(0)));
    assert_ne!(t, Tensor::from(vec![T(0)]));

    assert_eq!(t, t.clone().reshape([0]));
    assert_eq!(t.as_slice(), t.clone().as_slice());
    assert_eq!(t.clone().reshape([0]), t.clone().reshape([0]));

    assert_eq!(t.iter().len(), t.size());
    for _ in slice.iter() {
        assert!(false);
    }
    assert_eq!(t.get(0), None);

    for (a, b) in t.iter().zip(slice.iter()) {
        assert_eq!(a, b);
    }
}

#[test]
fn tensor_rank_3() {
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
fn debug_rank_0() {
    let t = Tensor::from(T(0));
    assert_eq!(format!("{:?}", t), "Tensor<essay_tensor::test::T> {T(0), shape: []}");
}

#[test]
fn debug_rank_1() {
    let v: [T; 0] = [];
    let t = Tensor::from(v);
    assert_eq!(format!("{:?}", t), "Tensor<essay_tensor::test::T> {[], shape: [0]}");

    let t = Tensor::from([T(0)]);
    assert_eq!(format!("{:?}", t), "Tensor<essay_tensor::test::T> {[T(0)], shape: [1]}");

    let t = Tensor::from([T(0), T(1)]);
    assert_eq!(format!("{:?}", t), "Tensor<essay_tensor::test::T> {[T(0) T(1)], shape: [2]}");

    let t = Tensor::from([1, 2, 3, 4, 5]);
    assert_eq!(format!("{:?}", t), "Tensor<i32> {[1 2 3 4 5], shape: [5]}");
}

#[test]
fn debug_rank_2() {
    let t = Tensor::from([[10]]);
    assert_eq!(format!("{:?}", t), "Tensor<i32> {\n[[10]], shape: [1, 1]}");

    let t = Tensor::from([[1, 2]]);
    assert_eq!(format!("{:?}", t), "Tensor<i32> {\n[[1 2]], shape: [1, 2]}");

    let t = Tensor::from([[1], [2]]);
    assert_eq!(format!("{:?}", t), "Tensor<i32> {\n[[1],\n [2]], shape: [2, 1]}");

    let t = Tensor::from([[1, 2, 3], [4, 5, 6]]);
    assert_eq!(format!("{:?}", t), "Tensor<i32> {\n[[1 2 3],\n [4 5 6]], shape: [2, 3]}");
}

#[test]
fn debug_rank_3() {
    let t = Tensor::from([
        [[10]]
    ]);
    assert_eq!(format!("{:?}", t), "Tensor<i32> {\n[[[10]]], shape: [1, 1, 1]}");

    let t = Tensor::from([
        [[1]],
        [[101]]
    ]);
    assert_eq!(format!("{:?}", t),  "Tensor<i32> {\n[[[1]],\n\n [[101]]], shape: [2, 1, 1]}");

    let t = Tensor::from([
        [[1], [101]],
    ]);
    assert_eq!(format!("{:?}", t),  "Tensor<i32> {\n[[[1],\n [101]]], shape: [1, 2, 1]}");

    let t = Tensor::from([
        [[1, 101]],
    ]);
    assert_eq!(format!("{:?}", t),  "Tensor<i32> {\n[[[1 101]]], shape: [1, 1, 2]}");

    let t = Tensor::from([
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
        [[101, 102, 103, 104], [105, 106, 107, 108], [109, 110, 111, 112]]
    ]);
    assert_eq!(format!("{:?}", t), "Tensor<i32> {
[[[1 2 3 4],
 [5 6 7 8],
 [9 10 11 12]],

 [[101 102 103 104],
 [105 106 107 108],
 [109 110 111 112]]], shape: [2, 3, 4]}");

    let t = Tensor::from([
        [[1, 2], [3, 4], [5, 6]],
        [[7, 8], [9, 10], [11, 12]],
        [[101, 102], [103, 104], [105, 106]],
        [[107, 108], [109, 110], [111, 112]]
    ]);
    assert_eq!(format!("{:?}", t), "Tensor<i32> {
[[[1 2],
 [3 4],
 [5 6]],

 [[7 8],
 [9 10],
 [11 12]],

 [[101 102],
 [103 104],
 [105 106]],

 [[107 108],
 [109 110],
 [111 112]]], shape: [4, 3, 2]}");
}

#[test]
fn debug_rank_4() {
    let t = Tensor::from([
        [
            [[10]]
        ]
    ]);
    assert_eq!(format!("{:?}", t), "Tensor<i32> {\n[[[[10]]]], shape: [1, 1, 1, 1]}");

    let t = Tensor::from([
        [
            [[10, 11]]
        ]
    ]);
    assert_eq!(format!("{:?}", t), "Tensor<i32> {\n[[[[10 11]]]], shape: [1, 1, 1, 2]}");

    let t = Tensor::from([
        [
            [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20]
            ],
            [
                [101, 102, 103, 104, 105],
                [106, 107, 108, 109, 110],
                [111, 112, 113, 114, 115],
                [116, 117, 118, 119, 120],
            ],
            [
                [201, 202, 203, 204, 205],
                [206, 207, 208, 209, 210],
                [211, 212, 213, 214, 215],
                [216, 217, 218, 219, 220],
            ]
        ],
        [
            [
                [1001, 1002, 1003, 1004, 1005],
                [1006, 1007, 1008, 1009, 1010],
                [1011, 1012, 1013, 1014, 1015],
                [1016, 1017, 1018, 1019, 1020]
            ],
            [
                [1101, 1102, 1103, 1104, 1105],
                [1106, 1107, 1108, 1109, 1110],
                [1111, 1112, 1113, 1114, 1115],
                [1116, 1117, 1118, 1119, 1120],
            ],
            [
                [1201, 1202, 1203, 1204, 1205],
                [1206, 1207, 1208, 1209, 1210],
                [1211, 1212, 1213, 1214, 1215],
                [1216, 1217, 1218, 1219, 1220],
            ]
        ]
    ]);
    assert_eq!(t.size(), t.shape().size());
    assert_eq!(format!("{:?}", t), "Tensor<i32> {
[[[[1 2 3 4 5],
 [6 7 8 9 10],
 [11 12 13 14 15],
 [16 17 18 19 20]],

 [[101 102 103 104 105],
 [106 107 108 109 110],
 [111 112 113 114 115],
 [116 117 118 119 120]],

 [[201 202 203 204 205],
 [206 207 208 209 210],
 [211 212 213 214 215],
 [216 217 218 219 220]]],

 [[[1001 1002 1003 1004 1005],
 [1006 1007 1008 1009 1010],
 [1011 1012 1013 1014 1015],
 [1016 1017 1018 1019 1020]],

 [[1101 1102 1103 1104 1105],
 [1106 1107 1108 1109 1110],
 [1111 1112 1113 1114 1115],
 [1116 1117 1118 1119 1120]],

 [[1201 1202 1203 1204 1205],
 [1206 1207 1208 1209 1210],
 [1211 1212 1213 1214 1215],
 [1216 1217 1218 1219 1220]]]], shape: [2, 3, 4, 5]}");
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
    assert_eq!(format!("{:?}", t), "Tensor<f32> {\n[[[1.0 2.0],\n [3.0 4.0]],\n\n [[11.0 12.0],\n [13.0 14.0]]], shape: [2, 2, 2]}");

    let t = ten![
        [[1., 2.], [3., 4.]],
        [[11., 12.], [13., 14.]],
        [[21., 22.], [23., 24.]]
    ];
    assert_eq!(format!("{:?}", t), "Tensor<f64> {\n[[[1.0 2.0],\n [3.0 4.0]],\n\n [[11.0 12.0],\n [13.0 14.0]],\n\n [[21.0 22.0],\n [23.0 24.0]]], shape: [3, 2, 2]}");
}

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
fn tensor_iter() {
    let vec : Vec<u32> = ten![1, 2, 3, 4].iter().map(|v| *v).collect();
    let vec2 : Vec<u32> = vec!(1, 2, 3, 4);
    assert_eq!(vec, vec2);

    let vec : Vec<u32> = ten![[1, 2], [3, 4]].iter().map(|v| *v).collect();
    let vec2 : Vec<u32> = vec!(1, 2, 3, 4);
    assert!(vec.iter().zip(vec2.iter()).all(|(x, y)| x == y));
}

#[test]
fn tensor_iter_slice() {
    let vec : Vec<Vec<u32>> = ten![1, 2, 3, 4].iter_row().map(|v| Vec::from(v)).collect();
    let vec2 : Vec<Vec<u32>> = vec!(vec!(1), vec!(2), vec!(3), vec!(4));
    assert_eq!(vec, vec2);

    let vec : Vec<Vec<u32>> = ten![[1, 2], [3, 4]].iter_row().map(|v| Vec::from(v)).collect();
    let vec2 : Vec<Vec<u32>> = vec!(vec![1, 2], vec![3, 4]);
    assert_eq!(vec, vec2);
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

    assert_eq!(t1, ten![1, 2, 3]);

    let t1: Tensor<i32> = [1, 2, 3].into_iter().collect();

    assert_eq!(t1, ten![1, 2, 3]);
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
