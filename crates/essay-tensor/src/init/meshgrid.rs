use crate::{Tensor, tensor::TensorUninit, prelude::Shape};

pub fn meshgrid<M: Meshgrid>(axes: M) -> M::Item {
    let vec = axes.build_meshgrid(false);
    /*
    let len = vec.len();
    if len >= 2 {
        vec.swap(len - 1, len - 2);
    }
    */

    M::to_output(&vec)
}

/// Note: separate function for 'ij' indexing
pub fn meshgrid_ij<M: Meshgrid>(axes: M) -> M::Item {
    let vec = axes.build_meshgrid(false);

    M::to_output(&vec)
}

fn build_meshgrid(axes: &[&Tensor], swap_xy: bool) -> Vec<Tensor> {
    let mut axes = Vec::from(axes);

    if swap_xy && axes.len() >= 2 {
        let len = axes.len();
        axes.swap(len - 1, len - 2);
    }
    let dims : Vec<usize> = axes.iter().map(|x| x.len()).collect();
    let shape = Shape::from(dims);

    let mut vec = Vec::<Tensor>::new();
    let mut inner = 1;

    for x in axes {
        vec.push(build_meshgrid_axis(x, &shape, inner));

        inner *= x.len();
    }

    vec
}

fn build_meshgrid_axis(x: &Tensor, shape: &Shape, i_s: usize) -> Tensor {
    assert!(x.rank() == 1);
    let size = shape.size();
    assert!(size % x.len() == 0);
    assert!(size % i_s == 0);

    unsafe {
        let len = x.len();
        let mut out = TensorUninit::<f32>::new(size);

        let x = x.as_slice();
        let o = out.as_mut_slice();

        let n = size / len;
        assert!(n % i_s == 0);

        let j_n = i_s;
        let i_n = n / i_s; // / k_s;
        let k_s = n / i_s;
        let j_s = len;

        for k in 0..len {
            let v = x[k];

            for j in 0..j_n {
                for i in 0..i_n {
                    o[k * k_s + j * j_s + i * i_s] = v;
                }
            }
        }

        out.into_tensor(shape)
    }
}

pub trait Meshgrid {
    type Item;

    fn build_meshgrid(&self, swap_xy: bool) -> Vec<Tensor>;
    fn to_output(vec: &Vec<Tensor>) -> Self::Item;
}

macro_rules! mesh_array {
    ($len: expr, $($i: expr),*) => {
        impl Meshgrid for [&Tensor; $len] {
            type Item = [Tensor; $len];
        
            fn build_meshgrid(&self, swap_xy: bool) -> Vec<Tensor> {
                build_meshgrid(self, swap_xy)
            }
        
            fn to_output(vec: &Vec<Tensor>) -> Self::Item {
                [$(vec[$i].clone()),*]
            }
        }

        impl Meshgrid for [Tensor; $len] {
            type Item = [Tensor; $len];
        
            fn build_meshgrid(&self, swap_xy: bool) -> Vec<Tensor> {
                let mut vec = Vec::<&Tensor>::new();

                for t in self {
                    vec.push(t);
                }

                build_meshgrid(vec.as_slice(), swap_xy)
            }
        
            fn to_output(vec: &Vec<Tensor>) -> Self::Item {
                [$(vec[$i].clone()),*]
            }
        }
    }
}

mesh_array!(0, );
mesh_array!(1, 0);
mesh_array!(2, 0, 1);

#[cfg(test)]
mod test {
    use crate::{init::linspace, tf32};

    use super::meshgrid;

    #[test]
    fn test_meshgrid_1d() {
        let [x] = meshgrid([&linspace(0., 3., 4)]);

        assert_eq!(x, tf32!([0., 1., 2., 3.]));
    }

    #[test]
    fn test_meshgrid_2d() {
        let [x, y] = meshgrid([
            &linspace(1., 3., 3),
            &linspace(10., 20., 2),
        ]);

        assert_eq!(x, tf32!([[1., 1.], [2., 2.], [3., 3.]]));
        assert_eq!(y, tf32!([[10., 20.], [10., 20.], [10., 20.]]));
    }
}