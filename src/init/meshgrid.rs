use crate::tensor::{Shape, Tensor, TensorData};

use super::linspace;

#[derive(Debug, PartialEq)]
pub enum Order {
    XY,
    IJ
}

/// Note: behavior differs from numpy for xyz (dim > 2) because the
/// numpy behavior doesn't make sense to me.
pub fn meshgrid<M: Meshgrid>(axes: M) -> M::Item {
    let vec = axes.build_meshgrid(Order::XY);

    M::to_output(&vec)
}

/// Note: separate function for 'ij' indexing
pub fn meshgrid_ij<M: Meshgrid>(axes: M) -> M::Item {
    let vec = axes.build_meshgrid(Order::IJ);

    M::to_output(&vec)
}

pub fn mgrid<const M: usize, const N: usize>() -> [Tensor; 2] {
    meshgrid([
        linspace(0., M as f32 - 1., M), 
        linspace(0., N as f32 - 1., N)
    ])
}

fn build_meshgrid(axes: &[&Tensor], order: Order) -> Vec<Tensor> {
    let mut axes = Vec::from(axes);

    if order == Order::XY { // && axes.len() >= 2 {
        // axes.swap(0, 1);
        axes.reverse();
    }

    let dims : Vec<usize> = axes.iter().map(|x| x.len()).collect();
    let shape = Shape::from(dims);

    //axes.reverse();

    let mut vec = Vec::<Tensor>::new();
    let mut inner = 1;

    for x in axes.iter().rev() {
        vec.push(build_meshgrid_axis(x, &shape, inner));

        inner *= x.len();
    }

    if order == Order::IJ {
        vec.reverse();
    }
    //if order == Order::XY && vec.len() >= 2 {
    //    vec.swap(0, 1)
    //}

    vec
}

fn build_meshgrid_axis(x: &Tensor, shape: &Shape, k_s: usize) -> Tensor {
    assert!(x.rank() == 1);
    let size = shape.size();
    assert!(size % x.len() == 0);
    assert!(size % k_s == 0);

    unsafe {
        let len = x.len();
        let n = size / len;
        assert!(n % k_s == 0);

        TensorData::<f32>::unsafe_init(size, |o| {
            let x = x.as_slice();

            let i_n = k_s;

            let j_n = n / i_n;
            let j_s = size / j_n;

            for k in 0..len {
                let v = x[k];

                for j in 0..j_n {
                    for i in 0..i_n {
                        o.add(k * k_s + j * j_s + i)
                            .write(v);
                    }
                }
            }
        }).into_tensor(shape)
    }
}

pub trait Meshgrid {
    type Item;

    fn build_meshgrid(&self, order: Order) -> Vec<Tensor>;
    fn to_output(vec: &Vec<Tensor>) -> Self::Item;
}

macro_rules! mesh_array {
    ($len: expr, $($i: expr),*) => {
        impl Meshgrid for [&Tensor; $len] {
            type Item = [Tensor; $len];
        
            fn build_meshgrid(&self, order: Order) -> Vec<Tensor> {
                build_meshgrid(self, order)
            }
        
            fn to_output(vec: &Vec<Tensor>) -> Self::Item {
                [$(vec[$i].clone()),*]
            }
        }

        impl Meshgrid for [Tensor; $len] {
            type Item = [Tensor; $len];
        
            fn build_meshgrid(&self, order: Order) -> Vec<Tensor> {
                let mut vec = Vec::<&Tensor>::new();

                for t in self {
                    vec.push(t);
                }

                build_meshgrid(vec.as_slice(), order)
            }
        
            fn to_output(vec: &Vec<Tensor>) -> Self::Item {
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
    use crate::{init::{linspace, meshgrid}, tf32};

    use super::meshgrid_ij;

    #[test]
    fn meshgrid_1d() {
        let [x] = meshgrid([&linspace(0., 3., 4)]);

        assert_eq!(x, tf32!([0., 1., 2., 3.]));
    }

    #[test]
    fn meshgrid_xy_2d() {
        let [x, y] = meshgrid([
            &linspace(0., 1., 2),
            &linspace(0., 2., 3),
            ]);

        assert_eq!(x, tf32!([[0., 1.], [0., 1.], [0., 1.]]));
        assert_eq!(y, tf32!([[0., 0.], [1., 1.], [2., 2.]]));
    }

    #[test]
    fn meshgrid_xyz_3d() {
        // Note difference in output from numpy
        let [x, y, z] = meshgrid([
            &linspace(0., 1., 2),
            &linspace(0., 2., 3),
            &linspace(0., 3., 4),
        ]);

        assert_eq!(x, tf32!([
            [[0., 1.], [0., 1.], [0., 1.]],
            [[0., 1.], [0., 1.], [0., 1.]],
            [[0., 1.], [0., 1.], [0., 1.]],
            [[0., 1.], [0., 1.], [0., 1.]],
        ]));
        assert_eq!(y, tf32!([
            [[0., 0.], [1., 1.], [2., 2.]],
            [[0., 0.], [1., 1.], [2., 2.]],
            [[0., 0.], [1., 1.], [2., 2.]],
            [[0., 0.], [1., 1.], [2., 2.]],
        ]));
        assert_eq!(z, tf32!([
            [[0., 0.], [0., 0.], [0., 0.]],
            [[1., 1.], [1., 1.], [1., 1.]],
            [[2., 2.], [2., 2.], [2., 2.]],
            [[3., 3.], [3., 3.], [3., 3.]],
        ]));
    }

    #[test]
    fn meshgrid_ij_1d() {
        let [i] = meshgrid_ij([&linspace(0., 3., 4)]);

        assert_eq!(i, tf32!([0., 1., 2., 3.]));
    }

    #[test]
    fn meshgrid_ij_2d() {
        let [j, i] = meshgrid_ij([
            &linspace(1., 3., 3),
            &linspace(10., 20., 2),
        ]);

        assert_eq!(i, tf32!([[10., 20.], [10., 20.], [10., 20.]]));
        assert_eq!(j, tf32!([[1., 1.], [2., 2.], [3., 3.]]));
    }

    #[test]
    fn meshgrid_ij_3d() {
        let [k, j, i] = meshgrid_ij([
            &linspace(0., 3., 4),
            &linspace(0., 2., 3),
            &linspace(0., 1., 2),
        ]);

        assert_eq!(i, tf32!([
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
        ]));

        assert_eq!(j, tf32!([
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
        ]));
            
        assert_eq!(k, tf32!([
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
        ]));
    }
}