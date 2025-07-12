use std::ptr;

use crate::tensor::scalar;

use super::{unsafe_init, Axis, Shape, Tensor, Type};

impl<T: Type> Tensor<T> {
    pub fn init<F>(shape: impl Into<Shape>, f: F) -> Self
    where
        F: FnMut() -> T
    {
        init(shape, f)
    }

    pub fn init_rindexed<F>(shape: impl Into<Shape>, f: F) -> Self
    where
        F: FnMut(&[usize]) -> T
    {
        init_rindexed(shape, f)
    }

    #[inline]
    pub fn map<U, F>(&self, f: F) -> Tensor<U>
    where
        U: Type,
        F: FnMut(&T) -> U
    {
        map(self.shape().clone(), self, f)
    }

    /// map_row is like map, but it iterates over the row, using
    /// each row as its argument and returning a new fixed-size row
    pub fn map_row<const N: usize, U: Type + Clone>(
        &self, 
        f: impl FnMut(&[T]) -> [U; N]
    ) -> Tensor<U> {
        let shape = if N == 1 && self.shape().rank() > 1 {
            self.shape().clone().rremove(0)
        } else {
            self.shape().clone().with_cols(N)
        };

        map_row(shape, self, f)
    }

    /// map_expand is like map, but returns a fixed-size slice that
    /// expands to fill columns in the new tensor.
    pub fn map_expand<const N: usize, U: Type + Clone>(
        &self, 
        f: impl FnMut(&T) -> [U; N]
    ) -> Tensor<U> {
        let shape = self.shape().clone().rinsert(0, N);

        map_expand(shape, self, f)
    }

    #[inline]
    pub fn map2<U: Type, F, V>(
        &self, 
        rhs: &Tensor<U>,
        f: F
    ) -> Tensor<V>
    where
        V: Type,
        F: FnMut(&T, &U) -> V
    {
        let shape = self.shape().broadcast_to(rhs.shape());

        map2(shape, &self, rhs, f)
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

    #[inline]
    pub fn map3<U: Type, V: Type, F, W>(
        &self, 
        b: &Tensor<U>,
        c: &Tensor<V>,
        f: F
    ) -> Tensor<W>
    where
        U: Type,
        V: Type,
        W: Type,
        F: FnMut(&T, &U, &V) -> W
    {
        let shape = self.shape().broadcast_to(b.shape()).broadcast_to(c.shape());

        map3(shape, &self, b, c, f)
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

    pub fn fold_row<const N: usize, S, F, V>(&self, axis: impl Into<Axis>, init: S, f: F) -> Tensor<V>
    where
        S: Clone + FoldState<Out=[V; N]>,
        F: FnMut(S, &[T]) -> S,
        V: Type,
    {
        fold_row(&self, axis, init, f)
    }

    pub fn normalize<S, S1, F, G, H, V>(
        &self, 
        axis: impl Into<Axis>, 
        init: S, 
        accum: F, 
        complete: G,
        norm: H,
    ) -> Tensor<V>
    where
        S: Clone,
        F: FnMut(S, &T) -> S,
        G: FnMut(S) -> S1,
        H: FnMut(&S1, &T) -> V,
        V: Type,
    {
        normalize(&self, axis, init, accum, complete, norm)
    }
}

impl<T: Type + Clone> Tensor<T> {
    pub fn reduce(&self, f: impl FnMut(T, T) -> T) -> Tensor<T> {
        reduce(self, f)
    }

    pub fn reduce_axis(&self, axis: impl Into<Axis>, f: impl FnMut(T, T) -> T) -> Tensor<T> {
        reduce_axis(self, axis, f)
    }
}

pub fn init<V: Type>(
    shape: impl Into<Shape>, 
    mut f: impl FnMut() -> V
) -> Tensor<V>
{
    let shape = shape.into();
    let size = shape.size();

    unsafe {
        unsafe_init::<V>(size, shape, |o| {
            for i in 0..size {
                o.add(i).write( (f)());
            }
        })
    }
}

pub fn init_rindexed<V: Type>(
    shape: impl Into<Shape>, 
    mut f: impl FnMut(&[usize]) -> V,
) -> Tensor<V>
{
    let shape = shape.into();
    let size = shape.size();

    unsafe {
        unsafe_init::<V>(size, shape.clone(), |o| {
            let mut vec = Vec::<usize>::new();
            vec.resize(shape.rank(), 0);
            let index = vec.as_mut_slice();

            for i in 0..size {
                o.add(i).write((f)(index));

                shape.next_index(index);
            }
        })
    }
}

fn map<U: Type, V: Type>(
    shape: Shape,
    a: &Tensor<U>,
    mut f: impl FnMut(&U) -> V
) -> Tensor::<V> {
    let len = a.size();
    
    unsafe {
        unsafe_init::<V>(len, shape, |o| {
            let a = a.as_slice();
        
            for i in 0..len {
                o.add(i).write((f)(&a[i]));
            }
        })
    }
}

fn map_row<const N: usize, U: Type, V: Type + Clone>(
    shape: Shape,
    a: &Tensor<U>, 
    mut f: impl FnMut(&[U]) -> [V; N]
) -> Tensor<V> {
    let a_cols = a.cols();
    let len = a.size() / a_cols;

    unsafe {
        unsafe_init::<V>(N * len, shape, |o| {
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
    shape: Shape,
    a: &Tensor<U>, 
    mut f: impl FnMut(&U) -> [V; N]
) -> Tensor<V> {
    let len = a.size();

    unsafe {
        unsafe_init::<V>(N * len, shape, |o| {
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
    shape: Shape,
    a: &Tensor<T>,
    b: &Tensor<U>,
    mut f: F
) -> Tensor<V>
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
        unsafe_init::<V>(size, shape, |o| {
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
    shape: Shape,
    a: &Tensor<T>,
    a_cols: usize,
    b: &Tensor<U>,
    b_cols: usize,
    mut f: F
) -> Tensor<V>
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
        unsafe_init::<V>(size, shape, |o| {
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

fn map3<T, U, V, W, F>(
    shape: Shape,
    a: &Tensor<T>,
    b: &Tensor<U>,
    c: &Tensor<V>,
    mut f: F
) -> Tensor<W>
where
    T: Type,
    U: Type,
    V: Type,
    W: Type,
    F: FnMut(&T, &U, &V) -> W
{
    let a_len = a.size();
    let b_len = b.size();
    let c_len = c.size();

    let size = a_len.max(b_len).max(c_len);
    let inner = a_len.min(b_len).min(c_len);
    let batch = size / inner;

    assert!(batch * inner == size, "broadcast mismatch a.len={} b.len={} c.len={}", a_len, b_len, c_len);
    
    unsafe {
        unsafe_init::<W>(size, shape, |o| {
            for n in 0..batch {
                let offset = n * inner;

                let a = a.as_wrap_slice(offset);
                let b = b.as_wrap_slice(offset);
                let c = c.as_wrap_slice(offset);

                for k in 0..inner {
                    o.add(offset + k).write(f(&a[k], &b[k], &c[k]));
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

    Tensor::from(value.into_result())
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
        unsafe_init::<V>(o_shape.size(), o_shape, |o| {
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
        })
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
        unsafe_init::<V>(o_shape.size(), o_shape, |o| {
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
        })
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
) -> Tensor<T> 
where
    T: Type + Clone,
    F: FnMut(T, T) -> T,
{
    let a = tensor.as_slice();

    let mut value = a[0].clone();

    for i in 1..tensor.size() {
        value = (f)(value, a[i].clone());
    }

    scalar(value)
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
        unsafe_init::<T>(o_shape.size(), o_shape, |o| {
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
        })
    }
}


pub(super) fn normalize<T, V, S, S1, F, G, H>(
    tensor: &Tensor<T>,
    axis: impl Into<Axis>,
    init: S,
    mut accum: F,
    mut complete: G,
    mut norm: H,
) -> Tensor<V> 
where
    T: Type,
    S: Clone,
    F: FnMut(S, &T) -> S,
    G: FnMut(S) -> S1,
    H: FnMut(&S1, &T) -> V,
    V: Type,
{
    let axis = axis.into();

    let (_o_shape, batch, a_len, inner) = axis.reduce(tensor.shape());
    // not axis.reduce, but normalize

    let shape = tensor.shape().clone();
    
    unsafe {
        unsafe_init::<V>(shape.size(), shape, |o| {
            let a = tensor.as_slice();

            for n in 0..batch {
                for i in 0..inner {
                    let mut state = init.clone();

                    for k in 0..a_len {
                        let v = &a[(n * a_len + k) * inner + i];

                        state = (accum)(state, v);
                    }

                    let state = (complete)(state);

                    for k in 0..a_len {
                        let v = &a[(n * a_len + k) * inner + i];

                        o.add((n * inner + i) * a_len + k)
                            .write((norm)(&state, v));
                    }
                }
            }
        })
    }
}

#[cfg(test)]
mod test {
    use crate::{ten, tensor::{scalar, Tensor}, test::{C, T, T2, T3, T4}};

    use super::FoldState;

    #[test]
    fn init() {
        let mut count = 10;
        let v = Tensor::init([3, 2], || {
            let v = T(count);
            count += 1;
            v
        });
    
        assert_eq!(v, ten![[T(10), T(11)], [T(12), T(13)], [T(14), T(15)]]);
        assert_eq!(v.shape(), &[3, 2].into());

        let v = Tensor::init_rindexed([3, 2], |idx| {
            T(100 * idx[1] + idx[0])
        });
    
        assert_eq!(v, ten![[T(0), T(1)], [T(100), T(101)], [T(200), T(201)]]);
        assert_eq!(v.shape(), &[3, 2].into());
    }

    #[test]
    fn map() {
        let a = ten![[T(1), T(20)], [T(3), T(40)], [T(5), T(60)]];
    
        let c = a.map(|a| T2(100 + a.0));
    
        assert_eq!(c, ten![[T2(101), T2(120)], [T2(103), T2(140)], [T2(105), T2(160)]]);
        assert_eq!(c.shape(), &[3, 2].into());
    }

    #[test]
    fn map2() {
        let a = ten![[T(1), T(20)], [T(3), T(40)], [T(5), T(60)]];
        let b = ten![[T2(100), T2(2000)], [T2(300), T2(4000)], [T2(500), T2(6000)]];
    
        let c = a.map2(&b, |a, b| T3(a.0 + b.0));
    
        assert_eq!(c, ten![[T3(101), T3(2020)], [T3(303), T3(4040)], [T3(505), T3(6060)]]);
    
        assert_eq!(c.shape(), &[3, 2].into());
    }

    #[test]
    fn map3() {
        let a = ten![[T(1), T(20)], [T(3), T(40)], [T(5), T(60)]];
        let b = ten![[T2(100), T2(2000)], [T2(300), T2(4000)], [T2(500), T2(6000)]];
        let c = ten![[T3(10000), T3(200000)], [T3(30000), T3(400000)], [T3(50000), T3(600000)]];
    
        let d = a.map3(&b, &c, |a, b, c| T4(a.0 + b.0 + c.0));
    
        assert_eq!(d, ten![[T4(10101), T4(202020)], [T4(30303), T4(404040)], [T4(50505), T4(606060)]]);
    
        assert_eq!(d.shape(), &[3, 2].into());
    }

    #[test]
    fn reduce() {
        let a = ten![[C(1), C(200)], [C(3), C(400)], [C(5), C(600)]];
        let v = a.reduce(|a, b| C(a.0 + b.0));
        assert_eq!(v, scalar(C(1209)));

        let a = ten![[C(1), C(200)], [C(3), C(400)], [C(5), C(600)]];
        let v = a.reduce_axis(None, |a, b| C(a.0 + b.0));
        assert_eq!(v, scalar(C(1209)));

        let v = a.reduce_axis(-1, |a, b| C(a.0 + b.0));
        assert_eq!(v, ten![C(201), C(403), C(605)]);

        let v = a.reduce_axis(0, |a, b| C(a.0 + b.0));
        assert_eq!(v, ten![C(9), C(1200)]);
    }

    #[test]
    fn fold() {
        let a = ten![[T(1), T(200)], [T(3), T(400)], [T(5), T(600)]];
        let v = a.fold(S(10000), |s, v| S(s.0 + v.0));
        assert_eq!(v, scalar(T2(11209)));

        let v = a.fold_axis(None, S(10000), |s, v| S(s.0 + v.0));
        assert_eq!(v, scalar(T2(11209)));

        let v = a.fold_axis(-1, S(10000), |s, v| S(s.0 + v.0));
        assert_eq!(v, ten![T2(10201), T2(10403), T2(10605)]);

        let v = a.fold_axis(0, S(10000), |s, v| S(s.0 + v.0));
        assert_eq!(v, ten![T2(10009), T2(11200)]);
    }

    #[derive(Clone, Debug)]
    struct S(usize);

    impl FoldState for S {
        type Out = T2;
    
        fn into_result(self) -> Self::Out {
            T2(self.0)
        }
    }
}