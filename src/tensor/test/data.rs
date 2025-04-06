use std::sync::{Arc, Mutex};

use crate::{
    ten, tensor::{data::TensorData, unsafe_init, Shape, Tensor, Type}, test::{Dead, Messages}
};

#[test]
#[should_panic]
fn from_boxed_slice_empty_panic() {
    let slice: [i32; 0] = [];
    let boxed_slice = Box::new(slice);

    TensorData::from_boxed_slice(boxed_slice, Shape::scalar());
}

#[test]
fn from_boxed_slice_scalar() {
    let slice: [u32; 1] = [0xaaaa5555];
    let boxed_slice = Box::new(slice);

    let tensor = TensorData::from_boxed_slice(boxed_slice, Shape::scalar());

    assert_eq!(tensor, Tensor::from(0xaaaa5555));
    assert_eq!(tensor.as_slice(), &[0xaaaa5555]);
    assert_eq!(tensor.shape().as_vec(), vec![]);
    assert_eq!(tensor.shape().size(), 1);
    assert_eq!(tensor.rank(), 0);
    assert_eq!(tensor.size(), 1);
    assert_eq!(tensor.offset(), 0);
}

#[test]
fn from_boxed_slice_drop() {
    Messages::clear();

    {
        let slice = [Dead(0x10), Dead(0x11)];
        let boxed_slice = Box::new(slice);
    
        {
            TensorData::from_boxed_slice(boxed_slice, 2);
        }

        assert_eq!(Messages::take(), vec!["Dead(10)", "Dead(11)"]);
    }

    assert_eq!(Messages::take(), Vec::<String>::new());
}

#[test]
#[should_panic]
fn from_boxed_rows_empty_panic() {
    let slice: [[i32; 4]; 0] = [];
    let boxed_slice = Box::new(slice);

    TensorData::from_boxed_rows(boxed_slice, Shape::scalar());
}

#[test]
fn from_boxed_rows_scalar() {
    let slice = [[0xaaaa5555u32], [0x5555aaaa]];
    let boxed_rows = Box::new(slice);

    let tensor = TensorData::from_boxed_rows(boxed_rows, [2, 1]);

    assert_eq!(tensor, Tensor::from([[0xaaaa5555], [0x5555aaaa]]));
    assert_eq!(tensor.as_slice(), &[0xaaaa5555, 0x5555aaaa]);
    assert_eq!(tensor.shape().as_vec(), vec![2, 1]);
    assert_eq!(tensor.shape().size(), 2);
    assert_eq!(tensor.rank(), 2);
    assert_eq!(tensor.size(), 2);
    assert_eq!(tensor.offset(), 0);
}

#[test]
fn from_boxed_rows_drop() {
    Messages::clear();

    {
        let slice = [[Dead(0x10), Dead(0x11)], [Dead(0x20), Dead(0x21)]];
        let boxed_rows = Box::new(slice);
    
        {
            TensorData::from_boxed_rows(boxed_rows, [2, 2]);
        }

        assert_eq!(Messages::take(), vec!["Dead(10)", "Dead(11)", "Dead(20)", "Dead(21)"]);
    }

    assert_eq!(Messages::take(), Vec::<String>::new());
}

#[test]
#[should_panic]
fn unsafe_init_empty_panic() {
    unsafe {
        unsafe_init::<u32>(0, Shape::scalar(), |_o| {
        });
    }
}

#[test]
fn unsafe_init_scalar() {
    let tensor = unsafe {
        unsafe_init::<i32>(1, Shape::scalar(), |o| {
            o.write(0x10);
        })
    };

    assert_eq!(tensor, Tensor::from(0x10));
    assert_eq!(tensor.as_slice(), &[0x10]);
    assert_eq!(tensor.shape().as_vec(), vec![]);
    assert_eq!(tensor.shape().size(), 2);
    assert_eq!(tensor.rank(), 2);
    assert_eq!(tensor.size(), 2);
    assert_eq!(tensor.offset(), 0);
}

#[test]
fn unsafe_init_tensor() {
    let tensor = unsafe {
        unsafe_init::<i32>(4, [2, 2], |o| {
            o.add(0).write(0x10);
            o.add(1).write(0x11);
            o.add(2).write(0x20);
            o.add(3).write(0x21);
        })
    };

    assert_eq!(tensor, ten![[0x10, 0x11], [0x20, 0x21]]);
    assert_eq!(tensor.as_slice(), &[0x10, 0x11, 0x20, 0x21]);
    assert_eq!(tensor.shape().as_vec(), vec![2, 2]);
    assert_eq!(tensor.shape().size(), 4);
    assert_eq!(tensor.rank(), 2);
    assert_eq!(tensor.size(), 4);
    assert_eq!(tensor.offset(), 0);
}

#[test]
fn unsafe_init_drop_scalar() {
    Messages::clear();

    unsafe {
        unsafe_init::<Dead>(1, Shape::scalar(), |o| {
            o.add(0).write(Dead(0x10));
        })
    };

    assert_eq!(Messages::take(), vec!["Dead(10)"]);
}

#[test]
fn unsafe_init_drop_matrix() {
    Messages::clear();

    unsafe {
        unsafe_init::<Dead>(4, [2, 2], |o| {
            o.add(0).write(Dead(0x10));
            o.add(1).write(Dead(0x11));
            o.add(2).write(Dead(0x20));
            o.add(3).write(Dead(0x21));
            // duplicate write doesn't cause a drop
            o.add(3).write(Dead(0x31));
        })
    };

    assert_eq!(Messages::take(), vec!["Dead(10)", "Dead(11)", "Dead(20)", "Dead(31)"]);
}

// older tests

#[test]
fn data_drop_from_vec() {
    let (p1, p2) = {
        let t1 = Test::new(1);
        let p1 = t1.ptr.clone();

        let t2 = Test::new(2);
        let p2 = t2.ptr.clone();

        let vec = vec![t1, t2];

        let _data = Tensor::<Test>::from_vec(vec, 2);

        (p1, p2)
    };

    assert_eq!(take(&p1), "Drop[1]");
    assert_eq!(take(&p2), "Drop[2]");
}

#[test]
fn data_drop_from_row_vec() {
    let (p1, p2) = {
        let t1 = Test::new(1);
        let p1 = t1.ptr.clone();

        let t2 = Test::new(2);
        let p2 = t2.ptr.clone();

        let vec = vec![[t1, t2]];

        let slice = vec.into_boxed_slice();

        let _data = TensorData::<Test>::from_boxed_rows(slice, [1, 2]);

        (p1, p2)
    };

    assert_eq!(take(&p1), "Drop[1]");
    assert_eq!(take(&p2), "Drop[2]");
}

#[test]
fn data_drop_from_boxed_slice() {
    let (p1, p2) = {
        let t1 = Test::new(1);
        let p1 = t1.ptr.clone();

        let t2 = Test::new(2);
        let p2 = t2.ptr.clone();

        let slice = Box::new([t1, t2]);

        let _data = TensorData::<Test>::from_boxed_slice(slice, [1, 2]);

        (p1, p2)
    };

    assert_eq!(take(&p1), "Drop[1]");
    assert_eq!(take(&p2), "Drop[2]");
}

#[test]
fn data_drop_from_slice() {
    let (p1, p2) = {
        let t1 = Test::new(1);
        let p1 = t1.ptr.clone();

        let t2 = Test::new(2);
        let p2 = t2.ptr.clone();

        let slice = [t1, t2];

        let _data = Tensor::<Test>::from_slice(&slice);

        (p1, p2)
    };

    assert_eq!(take(&p1), "Drop[1], Drop[1]");
    assert_eq!(take(&p2), "Drop[2], Drop[2]");
}

#[test]
fn drop_tensor() {
    let ptr = {
        let test = Test::new(2);
        let ptr = test.ptr.clone();
        let _tensor = Tensor::from(test);

        ptr
    };

    assert_eq!(take(&ptr), "Drop[2], Drop[2]");

    let ptr = {
        let vec = vec![Test::new(2)];
        let ptr = vec[0].ptr.clone();
        let _tensor = Tensor::from_vec(vec, Shape::from(1));

        ptr
    };

    assert_eq!(take(&ptr), "Drop[2]");
}

#[test]
fn test_drop_clone() {
    let ptr = {
        let vec = vec![Test::new(2)];
        let ptr = vec[0].ptr.clone();
        let _tensor = Tensor::from_vec(vec, Shape::from(1));
        let _tensor2 = _tensor.clone();

        ptr
    };

    assert_eq!(take(&ptr), "Drop[2]");
}

#[test]
fn test_vec_align() {
    let mut vec = Vec::<(u8, u32)>::new();

    vec.push((0x01, 0x4000_1000));
    vec.push((0x02, 0x5000_1001));
    vec.push((0x03, 0x6000_1002));
    vec.push((0x04, 0x7000_1003));

    unsafe {
        let len = vec.len();
        let tensor = Tensor::from_vec(vec, len);

        assert_eq!(*tensor.as_ptr().add(0), (0x01, 0x4000_1000));
        assert_eq!(*tensor.as_ptr().add(1), (0x02, 0x5000_1001));
        assert_eq!(*tensor.as_ptr().add(2), (0x03, 0x6000_1002));
        assert_eq!(*tensor.as_ptr().add(3), (0x04, 0x7000_1003));

        assert_eq!(
            tensor.as_slice(),
            &[
                (0x01, 0x4000_1000),
                (0x02, 0x5000_1001),
                (0x03, 0x6000_1002),
                (0x04, 0x7000_1003)
            ]
        );
    }

    let mut vec = Vec::<(u32, u8)>::new();

    vec.push((0x4000_1000, 0x01));
    vec.push((0x5000_1001, 0x02));
    vec.push((0x6000_1002, 0x03));
    vec.push((0x7000_1003, 0x04));

    unsafe {
        let len = vec.len();
        let tensor = Tensor::from_vec(vec, len);
        assert_eq!(*tensor.as_ptr().add(0), (0x4000_1000, 0x01));
        assert_eq!(*tensor.as_ptr().add(1), (0x5000_1001, 0x02));
        assert_eq!(*tensor.as_ptr().add(2), (0x6000_1002, 0x03));
        assert_eq!(*tensor.as_ptr().add(3), (0x7000_1003, 0x04));

        assert_eq!(
            tensor.as_slice(),
            &[
                (0x4000_1000, 0x01),
                (0x5000_1001, 0x02),
                (0x6000_1002, 0x03),
                (0x7000_1003, 0x04)
            ]
        );
    }
}

#[test]
fn test_vec_rows_basic() {
    let mut vec = Vec::<[u32; 3]>::new();

    vec.push([0x01, 0x1000, 0x01_0000]);
    vec.push([0x02, 0x2000, 0x02_0000]);
    vec.push([0x03, 0x3000, 0x03_0000]);
    vec.push([0x04, 0x4000, 0x04_0000]);

    let len = vec.len();
    let slice = vec.into_boxed_slice();
    let ten = TensorData::from_boxed_rows(slice, [len, 3]);
    assert_eq!(
        ten.as_slice(),
        &[
            0x01, 0x1000, 0x01_0000, 0x02, 0x2000, 0x02_0000, 0x03, 0x3000, 0x03_0000, 0x04,
            0x4000, 0x04_0000,
        ]
    );
}

#[test]
fn test_slice() {
    let a = ten!(10.);
    assert_eq!(a.as_slice(), &[10.]);

    let a = ten!([10.]);
    assert_eq!(a.as_slice(), &[10.]);

    let a = ten!([10., 20., 30.]);
    assert_eq!(a.as_slice(), &[10., 20., 30.]);

    let a = ten!([[10., 20.], [30., 40.]]);
    assert_eq!(a.as_slice(), &[10., 20., 30., 40.]);
}

#[test]
fn unsafe_init_dead() {
    Messages::clear();

    // TODO: validate the dropped values, which would require a static global
    unsafe {
        unsafe_init::<Dead>(1, [1], |o| {
            o.write(Dead(0x10));
            o.write(Dead(0x20));
            o.write(Dead(0x30));
        });
    }

    assert_eq!(Messages::take(), ["Dead(30)"]);

    unsafe {
        unsafe_init::<Dead>(1, [1], |o| {
            o.write(Dead(0x10));
            // Note: this drops the previous value, so can't be used
            *o = Dead(0x20);
            // Note: this does not drop the previous value
            o.write(Dead(0x30));
        });
    }

    assert_eq!(Messages::take(), ["Dead(10)", "Dead(30)"]);
}

#[test]
fn map_dead() {
    let t = ten![0x10, 0x20, 0x30];

    {
        t.map(|v| Dead(*v));
    }

    assert_eq!(Messages::take(), ["Dead(10)", "Dead(20)", "Dead(30)"]);
}

fn take(ptr: &Arc<Mutex<Vec<String>>>) -> String {
    let vec: Vec<String> = ptr.lock().unwrap().drain(..).collect();

    vec.join(", ")
}

#[derive(Debug, Clone)]
struct Test {
    id: usize,
    ptr: Arc<Mutex<Vec<String>>>,
}

impl Test {
    fn new(id: usize) -> Self {
        Self {
            id,
            ptr: Arc::new(Mutex::new(Vec::default())),
        }
    }
}

impl Type for Test {}

impl Drop for Test {
    fn drop(&mut self) {
        self.ptr
            .lock()
            .unwrap()
            .push(format!("Drop[{:?}]", self.id));
    }
}
