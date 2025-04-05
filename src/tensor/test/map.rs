use crate::{ten, tensor::{Axis, Type}};

#[test]
fn map_i32() {
    let t1 = ten![[1, 2, 3], [4, 5, 6]];
    let t2 = t1.map(|v| 10 + v);

    assert_eq!(t1.shape(), t2.shape());
    assert_eq!(t1.offset(), t2.offset());
    assert_eq!(t1.size(), t2.size());
    assert_eq!(t2, ten![[11, 12, 13], [14, 15, 16]]);
}

#[test]
fn map_i32_to_f32() {
    let t1 = ten![1, 2, 3, 4];
    let t2 = t1.map(|v| 2. * *v as f32);

    assert_eq!(t1.shape(), t2.shape());
    assert_eq!(t1.offset(), t2.offset());
    assert_eq!(t1.size(), t2.size());
    assert_eq!(t2, ten![2., 4., 6., 8.]);
}

#[test]
fn map_fnmut() {
    let t1 = ten![1, 2, 3, 4];
    let mut count = 1;
    let t2 = t1.map(|v| {
        let value = 10 * count + v;
        count += 1;
        value
    });

    assert_eq!(t2, ten![11, 22, 33, 44]);
}

#[test]
fn tensor_map_slice_4x1_to_4() {
    // when map_slice result dimension is 1, reduce the shape
    let t1 = ten![1, 2, 3, 4].reshape([4, 1]);
    let t2 = t1.map_row(|v| [2 * v[0]]);

    assert_eq!(t2.shape().as_vec(), &[4]);
    assert_eq!(t2, ten![2, 4, 6, 8]);
}

#[test]
fn map_row() {
    // when map_row result dimension is 1, reduce the shape
    let ten = ten![[1, 20], [3, 40], [5, 60]].map_row(|v| {
        [
            10 * v[0], 
            10 * v[1], 
            v[0] + v[1], 
            10 * (v[0] + v[1]),
        ]
    });

    assert_eq!(ten, ten![
        [10, 200, 21, 210],
        [30, 400, 43, 430],
        [50, 600, 65, 650],
    ]);
    assert_eq!(ten.shape(), &[3, 4].into());
    assert_eq!(ten.size(), 12);
    assert_eq!(ten.offset(), 0);
}

#[test]
fn map_row_combine() {
    // when map_slice result dimension is 1, reduce the shape
    let ten = ten![[1, 20], [3, 40], [5, 60]].map_row(|v| {
        [v[0] + 10 * v[1]]
    });

    assert_eq!(ten, ten![201, 403, 605]);
    assert_eq!(ten.shape(), &[3].into());
}

#[test]
fn tensor_map_expand_to_3() {
    let t1 = ten![1, 2, 3, 4].map_expand(|v| {
        [*v, *v * 10, *v * 100]
    });

    assert_eq!(t1, ten![
        [1, 10, 100],
        [2, 20, 200],
        [3, 30, 300],
        [4, 40, 400],
    ]);
}

#[test]
fn map2() {
    let a = ten![[1, 20], [3, 40], [5, 60]];
    let b = ten![[100, 2000], [300, 4000], [500, 6000]];

    let c = a.map2(&b, |a, b| a + b);

    assert_eq!(c, ten![[101, 2020], [303, 4040], [505, 6060]]);

    assert_eq!(c.shape(), &[3, 2].into());
    assert_eq!(c.size(), 6);
    assert_eq!(c.offset(), 0);
}

#[test]
fn map2_fnmut() {
    let a = ten![[1, 20], [3, 40], [5, 60]];
    let b = ten![[100, 2000], [300, 4000], [500, 6000]];

    let mut count  = 1;

    let c = a.map2(&b, |a, b| {
        let value = 10000 * count + a + b;
        count += 1;
        value
    });

    assert_eq!(c, ten![[10101, 22020], [30303, 44040], [50505, 66060]]);

    assert_eq!(c.shape(), &[3, 2].into());
    assert_eq!(c.size(), 6);
    assert_eq!(c.offset(), 0);
}

#[test]
fn map2_row() {
    let a = ten![[1, 2], [3, 4], [5, 6]];
    let b = ten![[10, 200], [30, 40], [50, 60]];

    let c = a.map2_row(&b, |a, b| [a[0] + b[0]]);

    assert_eq!(c, ten![[101, 2020], [303, 4040], [505, 6060]]);

    assert_eq!(c.shape(), &[3, 2].into());
    assert_eq!(c.size(), 6);
    assert_eq!(c.offset(), 0);
}

#[test]
fn fold_i32() {
    let t1 = ten![[1, 2, 3], [4, 5, 6]];
    let t2 = t1.fold(100, |s, v| s + v);

    assert_eq!(t2, ten![121]);
    assert_eq!(t2.shape(), &[1].into());
    assert_eq!(t2.size(), 1);
}

#[test]
fn fold_axis() {
    let t1 = ten![
        [[1, 2], [11, 12], [21, 22], [31, 32]],
        [[101, 102], [111, 112], [121, 122], [131, 132]],
        [[301, 302], [311, 312], [321, 322], [331, 332]],
    ];

    let t2 = t1.fold_axis(
        Axis::default(), 
        1000, 
        |s, v| s + v
    );
    
    assert_eq!(t2, ten![4596]);

    assert_eq!(t2.shape(), &[1].into());
    assert_eq!(t2.size(), 1);

    let t2 = t1.fold_axis(
        Axis::axis(-1), 
        1000, 
        |s, v| s + v
    );
    
    assert_eq!(t2, ten![
        [1003, 1023, 1043, 1063],
        [1203, 1223, 1243, 1263],
        [1603, 1623, 1643, 1663],
    ]);

    assert_eq!(t2.shape(), &[3, 4].into());
    assert_eq!(t2.size(), 12);

    let t2 = t1.fold_axis(
        Axis::axis(2), 
        1000, 
        |s, v| s + v
    );
    
    assert_eq!(t2, ten![
        [1003, 1023, 1043, 1063],
        [1203, 1223, 1243, 1263],
        [1603, 1623, 1643, 1663],
    ]);

    assert_eq!(t2.shape(), &[3, 4].into());
    assert_eq!(t2.size(), 12);

    let t2 = t1.fold_axis(
        Axis::axis(0), 
        1000, 
        |s, v| s + v
    );
    
    assert_eq!(t2, ten![
        [1403, 1406],
        [1433, 1436],
        [1463, 1466],
        [1493, 1496],
    ]);

    assert_eq!(t2.shape(), &[4, 2].into());
    assert_eq!(t2.size(), 8);

    let t2 = t1.fold_axis(
        Axis::axis(1), 
        1000, 
        |s, v| s + v
    );
    
    assert_eq!(t2, ten![
        [1064, 1068],
        [1464, 1468],
        [2264, 2268],
    ]);

    assert_eq!(t2.shape(), &[3, 2].into());
    assert_eq!(t2.size(), 6);
}

#[test]
fn fold_row_i32() {
    let t1 = ten![
        [[1, 2], [11, 12]],
    ];

    let t2 = t1.fold_row(None, 0, |s, v| {
        assert_eq!(v.len(), 2);
        s + 1000 * v[0] + v[1]
    });

    assert_eq!(t2, ten![12014]);
    assert_eq!(t2.shape(), &[1].into());
    assert_eq!(t2.size(), 1);

    let t1 = ten![
        [[1, 2], [10, 20], [100, 200]],
        [[3, 4], [30, 40], [300, 400]],
        [[5, 6], [50, 60], [500, 600]],
    ];

    let t2 = t1.fold_row(-2, 0, |s, v| {
        assert_eq!(v.len(), 2);
        s + 1000 * v[0] + v[1]
    });

    assert_eq!(t2, ten![111222, 333444, 555666]);
    assert_eq!(t2.shape(), &[3].into());
    assert_eq!(t2.size(), 3);
}

#[test]
fn reduce_shape() {
    // when map_slice result dimension is 1, reduce the shape
    let ten = ten![1, 2, 3, 4].reduce(|s, v| s + v);
    assert_eq!(ten, ten![10]);
    assert_eq!(ten.shape(), &[1].into());
    assert_eq!(ten.size(), 1);

    let ten = ten![[1, 2], [3, 4]].reduce(|s, v| s + v);
    assert_eq!(ten, ten![3, 7]);
    assert_eq!(ten.shape(), &[2].into());
    assert_eq!(ten.size(), 2);
}

#[test]
fn reduce_fnmut() {
    // when map_slice result dimension is 1, reduce the shape
    let mut count = 1;

    let ten = ten![[1, 2, 3], [4, 5, 6]].reduce(|s, v| {
        let value = s + v + count * 100;
        count += 1;
        value
    });
    assert_eq!(ten, ten![306, 715]);
}

/// reduce only depends on Clone + 'static
#[test]
fn reduce_with_clone() {
    let ten = ten![
        [ValueClone::from(1), ValueClone::from(2)], 
        [ValueClone::from(3), ValueClone::from(4)],
    ];

    let ten = ten.reduce(|s, t| ValueClone::from(s.value + t.value));
    assert_eq!(ten, ten![ValueClone::from(3), ValueClone::from(7)]);
}

#[derive(Clone, Debug, PartialEq)]
struct ValueClone {
    value: usize,
}

impl From<usize> for ValueClone {
    fn from(value: usize) -> Self {
        Self {
            value
        }
    }
}

impl Type for ValueClone {}
