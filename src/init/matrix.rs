use num_traits::{One, Zero};

use crate::tensor::{unsafe_init, Shape, Tensor, Type};

use super::linspace;

impl<T: Type + Zero + One> Tensor<T> {
    #[inline]
    pub fn eye(n: usize) -> Tensor<T> {
        eye(n)
    }

    #[inline]
    pub fn identity(n: usize) -> Tensor<T> {
        identity(n)
    }

    #[inline]
    pub fn tri(n: usize) -> Tensor<T> {
        tri(n)
    }
}

impl<T: Type + Clone + Zero> Tensor<T> {
    #[inline]
    pub fn diagflat(&self) -> Tensor<T> {
        diagflat(self)
    }

    #[inline]
    pub fn tril(&self) -> Tensor<T> {
        tril(self)
    }

    #[inline]
    pub fn triu(&self) -> Tensor<T> {
        triu(self)
    }
}

impl<T: Type + Clone> Tensor<T> {
    #[inline]
    pub fn meshgrid(&self, other: &Tensor<T>) -> [Tensor<T>; 2] {
        meshgrid([&self, other])
    }

    #[inline]
    pub fn meshgrid_ij(&self, other: &Tensor<T>) -> [Tensor<T>; 2] {
        meshgrid_ij([&self, other])
    }
}

pub fn eye<T: Type + Zero + One>(n: usize) -> Tensor<T> {
    if n == 0 {
        Tensor::from(T::zero())
    } else {
        Tensor::init_rindexed([n, n], |idx| {
            if idx[0] == idx[1] {
                T::one()
            } else {
                T::zero()
            }
        })
    }
}

#[inline]
pub fn identity<T>(n: usize) -> Tensor<T>
where
    T: Type + Zero + One
{
    eye(n)
}

pub fn diagflat<T>(diag: &Tensor<T>) -> Tensor<T> 
where
    T: Type + Clone + Zero
{
    assert!(diag.rank() == 1, "diagflat currently expects a 1d vector {:?}", diag.shape().as_vec());
    let n = diag.size();

    Tensor::init_rindexed([n, n], |idx| {
        if idx[0] == idx[1] {
            diag[idx[0]].clone()
        } else {
            T::zero()
        }
    })
}

pub fn tri<T>(n: usize) -> Tensor<T> 
where
    T: Type + Zero + One
{
    Tensor::init_rindexed([n, n], |idx| {
        if idx[0] <= idx[1] { T::one() } else { T::zero() }
    })
}

fn tril<T>(tensor: &Tensor<T>) -> Tensor<T>
where
    T: Type + Zero + Clone
{
    assert!(tensor.rank() >= 2);

    let size = tensor.shape().size();
    let shape = tensor.shape().clone();

    unsafe {
        unsafe_init::<T>(size, shape, |o| {
            let x = tensor.as_slice();

            let rows = tensor.rows();
            let cols = tensor.cols();
            let n = size / (rows * cols);

            for k in 0..n {
                for j in 0..rows {
                    for i in 0..cols {
                        let index = k * rows * cols + j * cols + i;

                        o.add(index)
                            .write(if i <= j { x[index].clone() } else { T::zero() });
                    }
                }
            }
        })
    }
}

pub fn triu<T>(tensor: &Tensor<T>) -> Tensor<T>
where
    T: Type + Clone + Zero
{
    assert!(tensor.rank() >= 2);

    let shape = tensor.shape().clone();
    let size = tensor.shape().size();

    unsafe {
        unsafe_init::<T>(size, shape, |o| {
            let x = tensor.as_slice();

            let rows = tensor.rows();
            let cols = tensor.cols();
            let n = size / (rows * cols);

            for k in 0..n {
                for j in 0..rows {
                    for i in 0..cols {
                        let index = k * rows * cols + j * cols + i;

                        o.add(index)
                            .write(if j <= i { x[index].clone() } else { T::zero() });
                    }
                }
            }
        })
    }
}

#[derive(Debug, PartialEq)]
pub enum Order {
    XY,
    IJ
}

/// Note: behavior differs from numpy for xyz (dim > 2) because the
/// numpy behavior doesn't make sense to me.
pub fn meshgrid<T, M: Meshgrid<T>>(axes: M) -> M::Item 
where
    T: Type + Clone
{
    axes.build(Order::XY)
}

/// Note: separate function for 'ij' indexing
pub fn meshgrid_ij<T, M: Meshgrid<T>>(axes: M) -> M::Item
where
    T: Type + Clone
{
    axes.build(Order::IJ)
}

pub fn mgrid<const M: usize, const N: usize>() -> [Tensor; 2] {
    meshgrid([
        linspace(0., M as f32 - 1., M), 
        linspace(0., N as f32 - 1., N)
    ])
}

fn build_meshgrid<T>(mut axes: Vec<&Tensor<T>>, order: Order) -> Vec<Tensor<T>>
where
    T: Type + Clone
{
    if order == Order::XY {
        // axes.swap(0, 1);
        axes.reverse();
    }

    let dims : Vec<usize> = axes.iter().map(|x| x.size()).collect();
    let shape = Shape::from(dims);

    let mut vec = Vec::new();
    let mut inner = 1;

    for x in axes.iter().rev() {
        vec.push(build_meshgrid_axis(x, &shape, inner));

        inner *= x.size();
    }

    if order == Order::IJ {
        vec.reverse();
    }

    vec
}

fn build_meshgrid_axis<T>(x: &Tensor<T>, shape: &Shape, k_s: usize) -> Tensor<T>
where
    T: Type + Clone
{
    assert!(x.rank() == 1);
    let size = shape.size();
    assert!(size % x.size() == 0);
    assert!(size % k_s == 0);

    unsafe {
        let len = x.size();
        let n = size / len;
        assert!(n % k_s == 0);

        unsafe_init::<T>(size, shape, |o| {
            let x = x.as_slice();

            let i_n = k_s;

            let j_n = n / i_n;
            let j_s = size / j_n;

            for k in 0..len {
                for j in 0..j_n {
                    for i in 0..i_n {
                        o.add(k * k_s + j * j_s + i)
                            .write(x[k].clone());
                    }
                }
            }
        })
    }
}

pub trait Meshgrid<T: Type + Clone> {
    type Item;

    fn build(&self, order: Order) -> Self::Item;
}

macro_rules! mesh_array {
    ($len: expr, $($i: expr),*) => {
        impl<T> Meshgrid<T> for [&Tensor<T>; $len]
        where
            T: Type + Clone
        {
            type Item = [Tensor<T>; $len];

            fn build(&self, order: Order) -> Self::Item {
                let vec = build_meshgrid(Vec::from(self), order);
                [$(vec[$i].clone()),*]
            }
        }

        impl<T> Meshgrid<T> for [Tensor<T>; $len]
        where
            T: Type + Clone
        {
            type Item = [Tensor<T>; $len];
        
            fn build(&self, order: Order) -> Self::Item {
                let mut vec = Vec::<&Tensor<T>>::new();

                for t in self {
                    vec.push(t);
                }

                let vec = build_meshgrid(vec, order);

                [$(vec[$i].clone()),*]
            }
        }
    }
}

// mesh_array!(0, );
mesh_array!(1, 0);
mesh_array!(2, 0, 1);
mesh_array!(3, 0, 1, 2);

#[cfg(test)]
mod test {
    use crate::{
        init::{eye, identity, linspace, matrix::{diagflat, tril, triu}, meshgrid, meshgrid_ij, tri}, 
        ten, tensor::Tensor, 
        test::{C, Z, ZO}
    };

    #[test]
    fn test_eye() {
        assert_eq!(eye(0), Tensor::from(0.));
        assert_eq!(eye(1), ten![[1.]]);
        assert_eq!(eye(2), ten![[1., 0.], [0., 1.]]);
        assert_eq!(eye(4), ten![
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ]);

        assert_eq!(
            eye(2), ten![
            [ZO(1), ZO(0)],
            [ZO(0), ZO(1)],
        ]);
    }

    #[test]
    fn test_identity() {
        assert_eq!(identity(0), Tensor::from(0.));
        assert_eq!(identity(1), ten![[1.]]);
        assert_eq!(identity(2), ten![[1., 0.], [0., 1.]]);
        assert_eq!(identity(4), ten![
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ]);

        assert_eq!(
            identity(2), ten![
            [ZO(1), ZO(0)],
            [ZO(0), ZO(1)],
        ]);
    }
    #[test]
    fn test_diagflat() {
        assert_eq!(
            diagflat(&ten![1., 2., 3.]), ten![
            [1., 0., 0.],
            [0., 2., 0.],
            [0., 0., 3.],
        ]);

        assert_eq!(
            diagflat(&ten![1, 2, 3]), ten![
            [1, 0, 0],
            [0, 2, 0],
            [0, 0, 3],
        ]);

        assert_eq!(
            diagflat(&ten![Z(1), Z(2)]), ten![
            [Z(1), Z(0)],
            [Z(0), Z(2)],
        ]);
    }
    #[test]
    fn test_tri() {
        assert_eq!(tri(1), ten![[1.]]);
        assert_eq!(tri(2), ten![
            [1., 0.],
            [1., 1.]
        ]);
        assert_eq!(tri(4), ten![
            [1., 0., 0., 0.],
            [1., 1., 0., 0.],
            [1., 1., 1., 0.],
            [1., 1., 1., 1.],
        ]);

        assert_eq!(tri(2), ten![
            [ZO(1), ZO(0)],
            [ZO(1), ZO(1)]
        ]);
    }

    #[test]
    fn test_tril() {
        assert_eq!(tril(&ten!([
            [1., 2., 3., 4.],
            [11., 12., 13., 14.],
            [21., 22., 23., 24.],
            [31., 32., 33., 34.],
        ])), ten!([
            [ 1.,  0.,  0.,  0.],
            [11., 12.,  0.,  0.],
            [21., 22., 23.,  0.],
            [31., 32., 33., 34.],
        ]));

        assert_eq!(tril(&ten!([
            [Z(1), Z(2)],
            [Z(11), Z(12)],
        ])), ten!([
            [Z(1), Z(0)],
            [Z(11), Z(12)],
        ]));
    }

    #[test]
    fn test_triu() {
        assert_eq!(triu(&ten![
            [1., 2., 3., 4.],
            [11., 12., 13., 14.],
            [21., 22., 23., 24.],
            [31., 32., 33., 34.],
        ]), ten![
            [1., 2., 3., 4.],
            [0., 12., 13., 14.],
            [0., 0., 23., 24.],
            [0., 0., 0., 34.],
        ]);

        assert_eq!(triu(&ten![
            [Z(1), Z(2)],
            [Z(11), Z(12)],
        ]), ten![
            [Z(1), Z(2)],
            [Z(0), Z(12)],
        ]);
    }
    #[test]
    fn meshgrid_1d() {
        let [x] = meshgrid([&linspace(0., 3., 4)]);
        assert_eq!(x, ten![0., 1., 2., 3.]);

        let [x] = meshgrid([&ten![C(1), C(2), C(3)]]);
        assert_eq!(x, ten![C(1), C(2), C(3)]);
    }

    #[test]
    fn meshgrid_xy_2d() {
        let [x, y] = meshgrid([
            &linspace(0., 1., 2),
            &linspace(0., 2., 3),
            ]);

        assert_eq!(x, ten![[0., 1.], [0., 1.], [0., 1.]]);
        assert_eq!(y, ten![[0., 0.], [1., 1.], [2., 2.]]);

        let [x, y] = meshgrid([
            &ten![C(0), C(1)],
            &ten![C(0), C(1), C(2)],
            ]);

        assert_eq!(x, ten![[C(0), C(1)], [C(0), C(1)], [C(0), C(1)]]);
        assert_eq!(y, ten![[C(0), C(0)], [C(1), C(1)], [C(2), C(2)]]);
    }

    #[test]
    fn meshgrid_xyz_3d() {
        // Note difference in output from numpy
        let [x, y, z] = meshgrid([
            &linspace(0., 1., 2),
            &linspace(0., 2., 3),
            &linspace(0., 3., 4),
        ]);

        assert_eq!(x, ten![
            [[0., 1.], [0., 1.], [0., 1.]],
            [[0., 1.], [0., 1.], [0., 1.]],
            [[0., 1.], [0., 1.], [0., 1.]],
            [[0., 1.], [0., 1.], [0., 1.]],
        ]);
        assert_eq!(y, ten![
            [[0., 0.], [1., 1.], [2., 2.]],
            [[0., 0.], [1., 1.], [2., 2.]],
            [[0., 0.], [1., 1.], [2., 2.]],
            [[0., 0.], [1., 1.], [2., 2.]],
        ]);
        assert_eq!(z, ten![
            [[0., 0.], [0., 0.], [0., 0.]],
            [[1., 1.], [1., 1.], [1., 1.]],
            [[2., 2.], [2., 2.], [2., 2.]],
            [[3., 3.], [3., 3.], [3., 3.]],
        ]);
    }

    #[test]
    fn meshgrid_ij_1d() {
        let [i] = meshgrid_ij([&linspace(0., 3., 4)]);
        assert_eq!(i, ten![0., 1., 2., 3.]);

        let [i] = meshgrid_ij([&ten![C(3), C(4), C(5)]]);
        assert_eq!(i, ten![C(3), C(4), C(5)]);
    }

    #[test]
    fn meshgrid_ij_2d() {
        let [j, i] = meshgrid_ij([
            &linspace(1., 3., 3),
            &linspace(10., 20., 2),
        ]);

        assert_eq!(i, ten![[10., 20.], [10., 20.], [10., 20.]]);
        assert_eq!(j, ten![[1., 1.], [2., 2.], [3., 3.]]);
    }

    #[test]
    fn meshgrid_ij_3d() {
        let [k, j, i] = meshgrid_ij([
            &linspace(0., 3., 4),
            &linspace(0., 2., 3),
            &linspace(0., 1., 2),
        ]);

        assert_eq!(i, ten![
            [[0., 1.],
            [0., 1.],
            [0., 1.]],

            [[0., 1.],
            [0., 1.],
            [0., 1.]],

            [[0., 1.],
            [0., 1.],
            [0., 1.]],

            [[0., 1.],
            [0., 1.],
            [0., 1.]],
        ]);

        assert_eq!(j, ten![
            [[0., 0.],
            [1., 1.],
            [2., 2.]],

            [[0., 0.],
            [1., 1.],
            [2., 2.]],

            [[0., 0.],
            [1., 1.],
            [2., 2.]],

            [[0., 0.],
            [1., 1.],
            [2., 2.]]
        ]);
            
        assert_eq!(k, ten![
            [[0., 0.],
            [0., 0.],
            [0., 0.]],

            [[1., 1.],
            [1., 1.],
            [1., 1.]],

            [[2., 2.],
            [2., 2.],
            [2., 2.]],

            [[3., 3.],
            [3., 3.],
            [3., 3.]],
        ]);
    }
}