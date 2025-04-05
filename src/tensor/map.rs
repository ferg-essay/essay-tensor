use std::ptr;

use super::{Axis, Shape, Tensor, TensorData, Type};

impl<T: Type> Tensor<T> {
    pub fn map<U, F>(&self, f: F) -> Tensor<U>
    where
        U: Type,
        F: FnMut(&T) -> U
    {
        map(self, f).into_tensor(self.shape())
    }

    // slice versions

    /// map_row is a map over a fixed-column tensor, such as a tensor of
    /// [f32; 2] pairs
    /// 
    pub fn map_row<const N: usize, U: Type + Clone>(
        &self, 
        f: impl FnMut(&[T]) -> [U; N]
    ) -> Tensor<U> {
        let shape = if N == 1 && self.shape().rank() > 1 {
            self.shape().clone().rremove(0)
        } else {
            self.shape().clone().with_cols(N)
        };

        map_row(self, f).into_tensor(shape)
    }

    // slice versions

    /// map_expand returns new columns
    /// 
    pub fn map_expand<const N: usize, U: Type + Clone>(
        &self, 
        f: impl FnMut(&T) -> [U; N]
    ) -> Tensor<U> {
        let shape = self.shape().clone().rinsert(0, N);

        map_expand(self, f).into_tensor(shape)
    }

    pub fn map2<U: Type, F, V: Type + 'static>(
        &self, 
        rhs: &Tensor<U>,
        f: F
    ) -> Tensor<V>
    where
        V: Type,
        F: FnMut(&T, &U) -> V
    {
        let shape = self.shape().broadcast_to(rhs.shape());

        map2(&self, rhs, f).into_tensor(shape)
    }

    pub fn map2_row<const N: usize, U, F, V>(
        &self, 
        _rhs: &Tensor<U>,
        _f: F
    ) -> Tensor<V>
    where
        U: Type,
        V: Type,
        F: FnMut(&[T], &[U]) -> [V; N]
    {
        todo!()
    }

    pub fn map2_expand<const N: usize, U, F, V: Clone + 'static>(
        &self, 
        _rhs: &Tensor<U>,
        _f: F
    ) -> Tensor<V>
    where
        U: Type,
        V: Clone + Type,
        F: FnMut(&T, &U) -> [V; N]
    {
        todo!()
    }

    pub fn fold<S, F, V>(&self, init: S, f: F) -> Tensor<V>
    where
        S: Clone + FoldState<Out=V>,
        F: FnMut(S, &T) -> S,
        V: Type,
    {
        fold(self, init, f)
    }

    pub fn fold_axis<S, F, V>(&self, axis: impl Into<Axis>, init: S, f: F) -> Tensor<V>
    where
        S: Clone + FoldState<Out=V>,
        F: FnMut(S, &T) -> S,
        V: Type,
    {
        fold_axis(self, axis, init, f)
    }

    pub fn fold_into<S, F, V>(&self, init: S, f: F) -> Tensor<V>
    where
        S: Clone + FoldState<Out=V>,
        F: FnMut(S, &T) -> S,
        V: Type,
    {
        fold(self, init, f)
    }

    pub fn fold_axis_into<S, F, V>(&self, axis: impl Into<Axis>, init: S, f: F) -> Tensor<V>
    where
        S: Clone + FoldState<Out=V>,
        F: FnMut(S, &T) -> S,
        V: Type,
    {
        fold_axis(self, axis, init, f)
    }

    pub fn fold_row<const N: usize, S, F, V>(&self, axis: impl Into<Axis>, init: S, f: F) -> Tensor<V>
    where
        S: Clone + FoldState<Out=[V; N]>,
        F: FnMut(S, &[T]) -> S,
        V: Type,
    {
        fold_row(&self, axis, init, f)
    }
}

impl<T: Type + Clone> Tensor<T> {
    pub fn reduce(&self, f: impl FnMut(T, T) -> T) -> Tensor<T> {
        let shape = if self.shape().rank() > 1 {
            self.shape().clone().rremove(0)
        } else {
            Shape::from([1])
        };

        reduce(self, f).into_tensor(shape)
    }

    pub fn reduce_axis(&self, axis: impl Into<Axis>, f: impl FnMut(T, T) -> T) -> Tensor<T> {
        reduce_axis(self, axis, f)
    }

    pub fn init<F>(shape: impl Into<Shape>, f: F) -> Self
    where
        F: FnMut() -> T
    {
        let shape = shape.into();

        Self::from_data(init(&shape, f), shape)
    }

    pub fn init_indexed<F>(shape: impl Into<Shape>, f: F) -> Self
    where
        F: FnMut(&[usize]) -> T
    {
        let shape = shape.into();

        Self::from_data(init_indexed(&shape, f), shape)
    }

    pub fn fill(shape: impl Into<Shape>, value: T) -> Self {
        Self::init(shape, || value.clone())
    }
}

fn init<F, V>(shape: impl Into<Shape>, mut f: F) -> TensorData<V>
where
    V: Type,
    F: FnMut() -> V,
{
    let shape = shape.into();
    let size = shape.size();

    unsafe {
        TensorData::<V>::unsafe_init(size, |o| {
            for i in 0..size {
                o.add(i).write( (f)());
            }
        }).into()
    }
}

fn init_indexed<F, V>(shape: impl Into<Shape>, mut f: F) -> TensorData<V>
where
    F: FnMut(&[usize]) -> V,
    V: Type
{
    let shape = shape.into();
    let size = shape.size();

    unsafe {
        TensorData::<V>::unsafe_init(size, |o| {
            let mut vec = Vec::<usize>::new();
            vec.resize(shape.rank(), 0);
            let index = vec.as_mut_slice();

            for i in 0..size {
                o.add(i).write((f)(index));

                shape.next_index(index);
            }
        }).into()
    }
}

fn map<U: Type, V: Type>(
    a: &Tensor<U>,
    mut f: impl FnMut(&U) -> V
) -> TensorData::<V> {
    let len = a.size();
    
    unsafe {
        TensorData::<V>::unsafe_init(len, |o| {
            let a = a.as_slice();
        
            for i in 0..len {
                o.add(i).write((f)(&a[i]));
            }
        })
    }
}

fn map_row<const N: usize, U: Type, V: Type + Clone>(
    a: &Tensor<U>, 
    mut f: impl FnMut(&[U]) -> [V; N]
) -> TensorData<V> {
    let a_cols = a.cols();
    let len = a.size() / a_cols;

    unsafe {
        TensorData::<V>::unsafe_init(N * len, |o| {
            let a = a.as_ptr();

            for i in 0..len {
                let slice = ptr::slice_from_raw_parts(a.add(i * a_cols), a_cols)
                    .as_ref()
                    .unwrap();

                let value = (f)(slice);

                for (j, value) in value.into_iter().enumerate() {
                    o.add(i * N + j).write(value);
                }
            }
        })
    }
}

fn map_expand<const N: usize, U: Type, V: Type>(
    a: &Tensor<U>, 
    mut f: impl FnMut(&U) -> [V; N]
) -> TensorData<V> {
    let len = a.size();

    unsafe {
        TensorData::<V>::unsafe_init(N * len, |o| {
            let a = a.as_slice();

            for i in 0..len {
                let value = (f)(&a[i]);

                for (j, value) in value.into_iter().enumerate() {
                    o.add(i * N + j).write(value);
                }
            }
        })
    }
}

fn map2<T, U, V, F>(
    a: &Tensor<T>,
    b: &Tensor<U>,
    mut f: F
) -> TensorData<V>
where
    T: Type,
    U: Type,
    V: Type,
    F: FnMut(&T, &U) -> V
{
    let a_len = a.size();
    let b_len = b.size();

    let size = a_len.max(b_len);
    let inner = a_len.min(b_len);
    let batch = size / inner;

    assert!(batch * inner == size, "broadcast mismatch a.len={} b.len={}", a_len, b_len);
    
    unsafe {
        TensorData::<V>::unsafe_init(size, |o| {
            for n in 0..batch {
                let offset = n * inner;

                let a = a.as_wrap_slice(offset);
                let b = b.as_wrap_slice(offset);

                for k in 0..inner {
                    o.add(offset + k).write(f(&a[k], &b[k]));
                }
            }
        })
    }
}

pub(super) fn _map2_row<const L: usize , const M: usize, const N: usize, T, U, V, F>(
    a: &Tensor<T>,
    a_cols: usize,
    b: &Tensor<U>,
    b_cols: usize,
    mut f: F
) -> TensorData<V>
where
    T: Type,
    U: Type,
    V: Type,
    F: FnMut(&[T], &[U]) -> [V; N]
{
    let a_size = a.size() / a_cols;
    assert!(a_size * a_cols == a.size());
    let b_size = b.size() / b_cols;
    assert!(b_size * b_cols == b.size());

    let size = a_size.max(b_size);
    let inner = a_size.min(b_size);
    let batch = size / inner;

    assert!(batch * inner == size, "broadcast mismatch a.size={} b.size={}", a_size, b_size);
    
    unsafe {
        TensorData::<V>::unsafe_init(size, |o| {
            for n in 0..batch {
                let offset = n * inner;

                for k in 0..inner {
                    let a = a.as_wrap_slice_n(a_cols * (offset + k), a_cols);
                    let b = b.as_wrap_slice_n(b_cols * (offset + k), b_cols);

                    let value = f(&a, &b);

                    for (i, v) in value.into_iter().enumerate() {
                        o.add(offset + k * N + i).write(v);
                    }
                }
            }
        })
    }
}

pub(super) fn fold<U, S, V, F>(
    tensor: &Tensor<U>,
    init: S,
    mut f: F,
) -> Tensor<V> 
where
    U: Type,
    S: Clone + FoldState<Out=V>,
    V: Type,
    F: FnMut(S, &U) -> S,
{
    let len = tensor.size();

    let a = tensor.as_slice();

    let mut value = init.clone();

    for i in 0..len {
        value = (f)(value, &a[i]);
    }

    TensorData::<V>::from_vec(vec![value.into_result()]).into_tensor(Shape::scalar())
}

pub(super) fn fold_axis<T, V, S, F>(
    tensor: &Tensor<T>,
    axis: impl Into<Axis>,
    init: S,
    mut f: F,
) -> Tensor<V> 
where
    T: Type,
    S: Clone + FoldState<Out=V>,
    F: FnMut(S, &T) -> S,
    V: Type,
{
    let axis = axis.into();

    let (o_shape, batch, a_len, inner) = axis.reduce(tensor.shape());

    unsafe {
        TensorData::<V>::unsafe_init(o_shape.size(), |o| {
            let a = tensor.as_slice();

            for n in 0..batch {
                for i in 0..inner {
                    let mut state = init.clone();

                    for k in 0..a_len {
                        let v = &a[(n * a_len + k) * inner + i];

                        state = (f)(state, v);
                    }

                    o.add(n * inner + i).write(state.into_result());
                }
            }
        }).into_tensor(o_shape)
    }
}

pub(super) fn fold_row<const N: usize, T, V, S, F>(
    tensor: &Tensor<T>,
    axis: impl Into<Axis>,
    init: S,
    mut f: F,
) -> Tensor<V> 
where
    T: Type,
    S: Clone + FoldState<Out=[V; N]>,
    F: FnMut(S, &[T]) -> S,
    V: Type,
{
    let axis = axis.into();

    let (o_shape, batch, a_len, inner) = axis.reduce_row(tensor.shape(), N);
    let cols = tensor.cols();

    unsafe {
        TensorData::<V>::unsafe_init(o_shape.size(), |o| {
            for b in 0..batch {
                for j in 0..inner {
                    let mut state = init.clone();

                    for k in 0..a_len {
                        let offset = (b * a_len + k) * inner + j;
                        let v = tensor.as_wrap_slice_n(cols * offset, cols);

                        state = (f)(state, v);
                    }

                    let value = state.into_result();

                    for (i, value) in value.into_iter().enumerate() {
                        o.add((b * inner + j) * N + i).write(value);
                    }
                }
            }
        }).into_tensor(o_shape)
    }
}

pub trait FoldState {
    type Out;

    fn into_result(self) -> Self::Out;
}

impl<T: Type> FoldState for T {
    type Out = T;
    
    fn into_result(self) -> Self::Out {
        self
    }
}

impl<const N: usize, T: Type> FoldState for [T; N] {
    type Out = [T; N];
    
    fn into_result(self) -> Self::Out {
        self
    }
}

pub(super) fn reduce<T, F>(
    tensor: &Tensor<T>,
    mut f: F,
) -> TensorData<T> 
where
    T: Type + Clone,
    F: FnMut(T, T) -> T,
{
    let cols = tensor.cols();
    let len = tensor.size() / cols;
    assert!(cols * len == tensor.size());
    
    unsafe {
        TensorData::<T>::unsafe_init(len, |o| {
            let a = tensor.as_slice();

            for j in 0..len {
                let offset = j * cols;

                let mut value = a[offset].clone();

                for i in 1..cols {
                    value = (f)(value, a[offset + i].clone());
                }

                o.add(j).write(value.into());
            }
        })
    }
}

pub(super) fn reduce_axis<T, F>(
    tensor: &Tensor<T>,
    axis: impl Into<Axis>,
    mut f: F,
) -> Tensor<T> 
where
    T: Type + Clone,
    F: FnMut(T, T) -> T,
{
    let axis = axis.into();

    let (o_shape, batch, a_len, inner) = axis.reduce(tensor.shape());

    unsafe {
        TensorData::<T>::unsafe_init(o_shape.size(), |o| {
            let a = tensor.as_slice();

            for n in 0..batch {
                for i in 0..inner {
                    let mut state = a[n * a_len * inner + i + 0 * inner].clone();

                    for k in 1..a_len {
                        let v = a[n * a_len * inner + i + k * inner].clone();

                        state = (f)(state, v);
                    }

                    o.add(n * inner + i).write(state);
                }
            }
        }).into_tensor(o_shape)
    }
}

